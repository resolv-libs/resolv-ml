""" TODO - doc """
# TODO - add multi-backend support for probability distributions
import keras
from keras import ops as k_ops
from tensorflow_probability import distributions as tfp_dist


@keras.saving.register_keras_serializable(package="Divergences", name="BetaDivergence")
class BetaDivergenceRegularizer(keras.Layer):

    def __init__(self,
                 divergence_layer: keras.Layer,
                 max_beta: float = 1.0,
                 beta_rate: float = 0.0,
                 free_bits: float = 0.0,
                 name: str = "beta_div",
                 **kwargs):
        super(BetaDivergenceRegularizer, self).__init__(name=name, **kwargs)
        if not 0.0 <= beta_rate < 1.0:
            raise ValueError("beta_rate must be between 0 and 1.")
        self._divergence_layer = divergence_layer
        self._free_bits = free_bits
        self._max_beta = max_beta
        self._beta_rate = beta_rate

    def build(self, input_shape):
        super().build(input_shape)
        self._divergence_layer.build(input_shape)

    def call(self,
             inputs,
             posterior: tfp_dist.Distribution,
             iterations=None,
             training: bool = False,
             evaluate: bool = False,
             **kwargs):
        if training or evaluate:
            div = self._divergence_layer(inputs, posterior=posterior, training=training)
            # Compute the cost according to free_bits
            free_nats = self._free_bits * k_ops.log(2.0)
            div_cost = k_ops.maximum(div - free_nats, 0)
            # Compute beta for beta-VAE
            div_beta = 1.0 if evaluate else (1.0 - k_ops.power(self._beta_rate, iterations or 1)) * self._max_beta
            # Compute KL across the batch and update trackers
            div_loss = div_beta * k_ops.mean(div_cost)
            div_loss_bits = div_loss / k_ops.log(2.0)
            return div_loss, div_loss_bits, div_beta
        else:
            return 0., 0., 0.

    def compute_output_shape(self, input_shape):
        return (3,)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "divergence_layer": keras.saving.serialize_keras_object(self._divergence_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        divergence_layer = keras.saving.deserialize_keras_object(config.pop("divergence_layer"))
        return cls(divergence_layer=divergence_layer, **config)


@keras.saving.register_keras_serializable(package="Divergences", name="KLDivergence")
class KLDivergence(keras.Layer):

    def __init__(self,
                 prior: tfp_dist.Distribution,
                 name: str = "gauss_kl_div",**kwargs):
        super(KLDivergence, self).__init__(name=name, **kwargs)
        self._prior = prior

    def call(self, inputs, posterior: tfp_dist.Distribution, training: bool = False, **kwargs):
        kl_loss = tfp_dist.kl_divergence(posterior, self._prior)
        return kl_loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)
