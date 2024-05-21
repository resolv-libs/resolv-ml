""" TODO - doc """
# TODO - add multi-backend support for probability distributions
import keras
from tensorflow_probability import distributions as tfp_dist


@keras.saving.register_keras_serializable(package="Divergences", name="KLDivergence")
class KLDivergence(keras.Layer):

    def __init__(self,
                 prior: tfp_dist.Distribution,
                 name: str = "gauss_kl_div",
                 **kwargs):
        super(KLDivergence, self).__init__(name=name, **kwargs)
        self._prior = prior

    def call(self, inputs, posterior: tfp_dist.Distribution, training: bool = False, **kwargs):
        kl_loss = tfp_dist.kl_divergence(posterior, self._prior)
        return kl_loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)
