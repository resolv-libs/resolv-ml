# TODO - DOC
# TODO - add multi-backend support for probability distributions
from typing import Any

import keras
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


class Inference(keras.Layer):

    def posterior_distribution(self, inputs, training: bool = False) -> tfd.Distribution:
        raise NotImplementedError("get_posterior_distribution must be implemented by subclasses.")

    def prior_distribution(self, training: bool = False) -> tfd.Distribution:
        raise NotImplementedError("get_prior_distribution must be implemented by subclasses.")

    def compute_output_shape(self, input_shape):
        return (2,)

    def call(self, inputs: Any, training: bool = False, **kwargs):
        posterior = self.posterior_distribution(inputs, training=training)
        prior = self.prior_distribution(training=training)
        return posterior, prior


@keras.saving.register_keras_serializable(package="Inference", name="GaussianInference")
class GaussianInference(Inference):

    def __init__(self,
                 z_size: int,
                 mean_layer: keras.Layer = None,
                 sigma_layer: keras.Layer = None,
                 name: str = "gaussian_inference",
                 **kwargs):
        super(GaussianInference, self).__init__(name=name, **kwargs)
        self._z_size = z_size
        self._mean_layer = mean_layer if mean_layer else self.default_mean_layer(z_size)
        self._sigma_layer = sigma_layer if sigma_layer else self.default_sigma_layer(z_size)

    def build(self, input_shape):
        if not self._mean_layer.built:
            self._mean_layer.build(input_shape)
        if not self._sigma_layer.built:
            self._sigma_layer.build(input_shape)

    def posterior_distribution(self, inputs, training: bool = False) -> tfd.Distribution:
        vae_inputs, _ = inputs
        z_mean = self._mean_layer(vae_inputs)
        z_sigma = self._sigma_layer(vae_inputs)
        return tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=z_sigma)

    def prior_distribution(self, training: bool = False) -> tfd.Distribution:
        return tfd.MultivariateNormalDiag(loc=keras.ops.zeros(self._z_size), scale_diag=keras.ops.ones(self._z_size))

    @classmethod
    def default_mean_layer(cls, z_size: int) -> keras.layers.Dense:
        return keras.layers.Dense(
            units=z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name="z_mean"
        )

    @classmethod
    def default_sigma_layer(cls, z_size: int) -> keras.layers.Dense:
        return keras.layers.Dense(
            units=z_size,
            activation="softplus",
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            bias_initializer="zeros",
            name="z_sigma"
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_size": self._z_size,
            "mean_layer": keras.saving.serialize_keras_object(self._mean_layer),
            "sigma_layer": keras.saving.serialize_keras_object(self._sigma_layer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        mean_layer = keras.saving.deserialize_keras_object(config.pop("mean_layer"))
        sigma_layer = keras.saving.deserialize_keras_object(config.pop("sigma_layer"))
        return cls(mean_layer=mean_layer, sigma_layer=sigma_layer, **config)


@keras.saving.register_keras_serializable(package="Inference", name="NormalizingFlowGaussianInference")
class NormalizingFlowGaussianInference(GaussianInference):

    def __init__(self,
                 z_size: int,
                 normalizing_flow: tfb.Bijector,
                 target_dimension: int = 0,
                 mean_layer: keras.Layer = None,
                 sigma_layer: keras.Layer = None,
                 name: str = "nf_gaussian_inference",
                 **kwargs):
        if not 0 <= target_dimension < z_size:
            raise ValueError("Normalizing flow target dimension must be in the interval [0, z_size).")
        super(NormalizingFlowGaussianInference, self).__init__(
            z_size=z_size,
            mean_layer=mean_layer,
            sigma_layer=sigma_layer,
            name=name,
            **kwargs
        )
        self._normalizing_flow = normalizing_flow
        self._target_dimension = target_dimension

    def posterior_distribution(self, inputs, training: bool = False) -> tfd.Distribution:
        vae_inputs, aux_inputs = inputs
        z_mean = self._mean_layer(vae_inputs)
        z_sigma = self._sigma_layer(vae_inputs)
        return self._get_sequential_joint_distribution(z_mean, z_sigma)

    def prior_distribution(self, training: bool = False) -> tfd.Distribution:
        return self._get_sequential_joint_distribution()

    def _get_sequential_joint_distribution(self, z_mean=None, z_sigma=None):
        prior_z_mean = keras.ops.zeros(self._z_size)
        prior_z_sigma = keras.ops.ones(self._z_size)
        is_post = z_mean is not None and z_sigma is not None
        # Build the normalizing flow for the target dimension
        normalizing_flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
                loc=keras.ops.expand_dims(
                    z_mean[:, self._target_dimension] if is_post else prior_z_mean[self._target_dimension],
                    axis=-1
                ),
                scale_diag=keras.ops.expand_dims(
                    z_sigma[:, self._target_dimension] if is_post else prior_z_sigma[self._target_dimension],
                    axis=-1
                )
            ),
            bijector=self._normalizing_flow
        )
        # Split the original distribution into three parts: before target dim, at target dim, and after target dim
        before_target_dimension = tfd.MultivariateNormalDiag(
            loc=z_mean[:, :self._target_dimension] if is_post else prior_z_mean[:self._target_dimension],
            scale_diag=z_sigma[:, :self._target_dimension] if is_post else prior_z_sigma[:self._target_dimension],
        ) if self._target_dimension > 0 else None
        after_target_dimension = tfd.MultivariateNormalDiag(
            loc=z_mean[:, self._target_dimension + 1:] if is_post else prior_z_mean[self._target_dimension + 1:],
            scale_diag=z_sigma[:, self._target_dimension + 1:] if is_post else prior_z_sigma[
                                                                               self._target_dimension + 1:],
        ) if self._target_dimension < self._z_size - 1 else None
        components = []
        if before_target_dimension is not None:
            components.append(before_target_dimension)
        components.append(normalizing_flow)
        if after_target_dimension is not None:
            components.append(after_target_dimension)
        return tfd.JointDistributionSequential(components)


@keras.saving.register_keras_serializable(package="Inference", name="SamplingLayer")
class SamplingLayer(keras.Layer):

    def __init__(self,
                 z_size: int,
                 name: str = "sampling",
                 **kwargs):
        super(SamplingLayer, self).__init__(name=name, **kwargs)
        self._z_size = z_size

    def compute_output_shape(self, input_shape, **kwargs):
        return input_shape[0], self._z_size

    def call(self,
             inputs,
             prior: tfd.Distribution,
             posterior: tfd.Distribution = None,
             training: bool = False,
             evaluate: bool = False,
             **kwargs):
        if training or evaluate:
            z = posterior.sample()
            if isinstance(posterior, tfd.JointDistribution):
                z = keras.ops.concatenate(z, axis=-1)
        else:
            num_sequences = inputs
            z = prior.sample(sample_shape=(num_sequences,))
        return z


if __name__ == "__main__":
    from tensorflow_probability import bijectors as tfb
    from tensorflow_probability import distributions as tfd
    from resolv_ml.models.dlvm.normalizing_flows.power_transform import power_transform_bijector

    mean = tf.random.normal(shape=(32, 256))
    sigma = tf.random.normal(shape=(32, 256))
    nf = tfb.RealNVP(
        fraction_masked=0.5,
        shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=[512, 512])
    )
    # nf = power_transform_bijector()
    post_transformed_distribution = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(
            loc=tf.expand_dims(mean[:, 0], axis=-1),
            scale_diag=tf.expand_dims(sigma[:, 0], axis=-1)
        ),
        bijector=nf
    )
    post_multivariate_normal = tfd.MultivariateNormalDiag(
        loc=mean[:, 1:],
        scale_diag=sigma[:, 1:]
    )
    joint_post = tfd.JointDistributionSequential([post_transformed_distribution, post_multivariate_normal])

    prior_transformed_distribution = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(loc=[0]),
        bijector=nf
    )
    prior_multivariate_normal = tfd.MultivariateNormalDiag(
        loc=tf.zeros(shape=(255,)),
        scale_diag=tf.ones(shape=(255,))
    )
    joint_prior = tfd.JointDistributionSequential([prior_transformed_distribution, prior_multivariate_normal])
    samples = joint_post.sample()
    log_prob = joint_post.log_prob(samples)
    kld = tfd.kl_divergence(joint_prior, joint_post)
    pass
