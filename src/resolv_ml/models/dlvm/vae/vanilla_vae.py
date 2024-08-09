# TODO - DOC
# TODO - add multi-backend support for probability distributions

import keras
from tensorflow_probability import distributions as tfd

from .base import VAE
from ....utilities.distributions.inference import GaussianInference, SamplingLayer
from ....utilities.regularizers.divergence import DivergenceRegularizer
from ....utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="VAE", name="StandardVAE")
class StandardVAE(VAE):

    def __init__(self,
                 z_size: int,
                 input_processing_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 inference_layer: keras.Layer = None,
                 aux_input_processing_layer: keras.Layer = None,
                 div_beta_scheduler: Scheduler = None,
                 free_bits: float = 0.0,
                 name: str = "standard_vae",
                 **kwargs):
        self._z_size = z_size
        self._div_beta_scheduler = div_beta_scheduler
        self._free_bits = free_bits
        super(StandardVAE, self).__init__(
            input_processing_layer=input_processing_layer,
            generative_layer=generative_layer,
            inference_layer=GaussianInference(
                z_size=z_size,
                name="gaussian_inference"
            ) if not inference_layer else inference_layer,
            sampling_layer=SamplingLayer(z_size=z_size),
            aux_input_processing_layer=aux_input_processing_layer,
            regularizers={
                "kld": DivergenceRegularizer(
                    divergence_fn=tfd.kl_divergence,
                    weight_scheduler=div_beta_scheduler,
                    free_bits=free_bits,
                    name="kld"
                )
            },
            name=name,
            **kwargs
        )

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_size": self._z_size,
            "div_beta_scheduler": keras.saving.serialize_keras_object(self._div_beta_scheduler),
            "free_bits": self._free_bits,
            "input_processing_layer": keras.saving.serialize_keras_object(self._input_processing_layer),
            "generative_layer": keras.saving.serialize_keras_object(self._generative_layer),
            "inference_layer": keras.saving.serialize_keras_object(self._inference_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_processing_layer = keras.saving.deserialize_keras_object(config.pop("input_processing_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        inference_layer = keras.saving.deserialize_keras_object(config.pop("inference_layer"))
        div_beta_scheduler = keras.saving.deserialize_keras_object(config.pop("div_beta_scheduler"))
        return cls(
            input_processing_layer=input_processing_layer,
            generative_layer=generative_layer,
            inference_layer=inference_layer,
            div_beta_scheduler=div_beta_scheduler,
            **config
        )
