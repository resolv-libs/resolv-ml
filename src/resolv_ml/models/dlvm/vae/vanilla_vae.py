# TODO - DOC
# TODO - add multi-backend support for probability distributions

import keras
from tensorflow_probability import distributions as tfp_dist

from .base import VAE
from ....utilities.distributions.divergence import KLDivergence
from ....utilities.distributions.inference import GaussianInference, SamplingLayer
from ....utilities.regularizers.divergence import DivergenceRegularizer
from ....utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="VAE", name="StandardVAE")
class StandardVAE(VAE):

    def __init__(self,
                 z_size: int,
                 input_processing_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 aux_input_processing_layer: keras.Layer = None,
                 mean_inference_layer: keras.Layer = None,
                 sigma_inference_layer: keras.Layer = None,
                 div_beta_scheduler: Scheduler = None,
                 free_bits: float = 0.0,
                 name: str = "standard_vae",
                 **kwargs):
        self._z_size = z_size
        self._div_beta_scheduler = div_beta_scheduler
        self._free_bits = free_bits
        self._mean_inference_layer = mean_inference_layer
        self._sigma_inference_layer = sigma_inference_layer
        self._kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self._kl_beta_tracker = keras.metrics.Mean(name="kl_beta")
        self._kl_loss_weighted_tracker = keras.metrics.Mean(name="kl_loss_weighted")
        prior = tfp_dist.MultivariateNormalDiag(loc=keras.ops.zeros(z_size), scale_diag=keras.ops.ones(z_size))
        super(StandardVAE, self).__init__(
            input_processing_layer=input_processing_layer,
            generative_layer=generative_layer,
            inference_layer=GaussianInference(
                z_size=z_size,
                mean_layer=mean_inference_layer,
                sigma_layer=sigma_inference_layer,
                name="gaussian_inference",
            ),
            sampling_layer=SamplingLayer(z_size=z_size, prior=prior),
            aux_input_processing_layer=aux_input_processing_layer,
            regularization_layers={
                "kld": DivergenceRegularizer(
                    divergence_layer=KLDivergence(prior=prior),
                    beta_scheduler=div_beta_scheduler,
                    free_bits=free_bits
                )
            },
            name=name,
            **kwargs
        )

    def build(self, input_shape):
        super().build(input_shape)

    def _add_regularization_losses(self, regularization_losses):
        kl_loss, kl_beta = regularization_losses[0]
        weighted_kl_loss = kl_beta * kl_loss
        self.add_loss(weighted_kl_loss)
        self._kl_loss_tracker.update_state(kl_loss)
        self._kl_beta_tracker.update_state(kl_beta)
        self._kl_loss_weighted_tracker.update_state(weighted_kl_loss)
        
    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_size": self._z_size,
            "div_beta_scheduler": keras.saving.serialize_keras_object(self._div_beta_scheduler),
            "free_bits": self._free_bits,
            "input_processing_layer": keras.saving.serialize_keras_object(self._input_processing_layer),
            "mean_inference_layer": keras.saving.serialize_keras_object(self._mean_inference_layer),
            "sigma_inference_layer": keras.saving.serialize_keras_object(self._sigma_inference_layer),
            "generative_layer": keras.saving.serialize_keras_object(self._generative_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_processing_layer = keras.saving.deserialize_keras_object(config.pop("input_processing_layer"))
        mean_inference_layer = keras.saving.deserialize_keras_object(config.pop("mean_inference_layer"))
        sigma_inference_layer = keras.saving.deserialize_keras_object(config.pop("sigma_inference_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        div_beta_scheduler = keras.saving.deserialize_keras_object(config.pop("div_beta_scheduler"))
        return cls(
            input_processing_layer=input_processing_layer,
            mean_inference_layer=mean_inference_layer,
            sigma_inference_layer=sigma_inference_layer,
            generative_layer=generative_layer,
            div_beta_scheduler=div_beta_scheduler,
            **config
        )
