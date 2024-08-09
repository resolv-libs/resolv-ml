# TODO - DOC
# TODO - add multi-backend support for probability distributions

import keras
from tensorflow_probability import distributions as tfd


class Inference(keras.Layer):

    def posterior_distribution(self, inputs, training: bool = False) -> tfd.Distribution:
        raise NotImplementedError("get_posterior_distribution must be implemented by subclasses.")

    def prior_distribution(self, training: bool = False) -> tfd.Distribution:
        raise NotImplementedError("get_prior_distribution must be implemented by subclasses.")

    def compute_output_shape(self, input_shape):
        return (2,)

    def call(self, inputs, training: bool = False, **kwargs):
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
        else:
            num_sequences = inputs
            z = prior.sample(sample_shape=(num_sequences,))
        return z
