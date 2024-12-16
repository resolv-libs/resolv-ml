# TODO - DOC
# TODO - add multi-backend support for probability distributions

import keras
from tensorflow_probability import distributions as tfd

from .base import VAE
from ....utilities.distributions.inference import GaussianInference
from ....utilities.regularizers.divergence import DivergenceRegularizer
from ....utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="VAE", name="StandardVAE")
class StandardVAE(VAE):

    def __init__(self,
                 z_size: int,
                 feature_extraction_layer: keras.Layer,
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
            feature_extraction_layer=feature_extraction_layer,
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

    def get_latent_space_shape(self):
        return (self._z_size,)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_size": self._z_size,
            "div_beta_scheduler": keras.saving.serialize_keras_object(self._div_beta_scheduler),
            "free_bits": self._free_bits,
            "feature_extraction_layer": keras.saving.serialize_keras_object(self._feature_extraction_layer),
            "generative_layer": keras.saving.serialize_keras_object(self._generative_layer),
            "inference_layer": keras.saving.serialize_keras_object(self._inference_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        feature_extraction_layer = keras.saving.deserialize_keras_object(config.pop("feature_extraction_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        inference_layer = keras.saving.deserialize_keras_object(config.pop("inference_layer"))
        div_beta_scheduler = keras.saving.deserialize_keras_object(config.pop("div_beta_scheduler"))
        return cls(
            feature_extraction_layer=feature_extraction_layer,
            generative_layer=generative_layer,
            inference_layer=inference_layer,
            div_beta_scheduler=div_beta_scheduler,
            **config
        )


@keras.saving.register_keras_serializable(package="VAE", name="SamplingLayer")
class SamplingLayer(keras.Layer):

    def __init__(self,
                 z_size: int,
                 name: str = "sampling",
                 **kwargs):
        super(SamplingLayer, self).__init__(name=name, **kwargs)
        self._z_size = z_size

    def compute_output_shape(self, input_shape, **kwargs):
        vae_input_shape, _, _ = input_shape
        return vae_input_shape[0], self._z_size

    def call(self,
             inputs,
             prior: tfd.Distribution,
             posterior: tfd.Distribution = None,
             training: bool = False,
             evaluate: bool = False,
             **kwargs):
        return posterior.sample() if training or evaluate else prior.sample(sample_shape=(inputs,))
