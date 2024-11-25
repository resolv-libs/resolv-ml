# TODO - DOC
from typing import Dict

import keras

from .vanilla_vae import StandardVAE
from ....utilities.regularizers.attribute import AttributeRegularizer
from ....utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="VAE", name="AttributeRegularizedVAE")
class AttributeRegularizedVAE(StandardVAE):

    def __init__(self,
                 z_size: int,
                 feature_extraction_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 attribute_processing_layer: keras.Layer,
                 attribute_regularizers: Dict[str, AttributeRegularizer],
                 inference_layer: keras.Layer = None,
                 div_beta_scheduler: Scheduler = None,
                 free_bits: float = 0.0,
                 name: str = "ar_vae",
                 **kwargs):
        self._attribute_processing_layer = attribute_processing_layer
        self._attribute_regularizers = attribute_regularizers
        super(AttributeRegularizedVAE, self).__init__(
            z_size=z_size,
            feature_extraction_layer=feature_extraction_layer,
            aux_input_processing_layer=attribute_processing_layer,
            generative_layer=generative_layer,
            inference_layer=inference_layer,
            div_beta_scheduler=div_beta_scheduler,
            free_bits=free_bits,
            name=name,
            **kwargs
        )
        self._regularizers.update(attribute_regularizers)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "attribute_processing_layer": keras.saving.serialize_keras_object(self._attribute_processing_layer),
            "attribute_regularizers": keras.saving.serialize_keras_object(self._attribute_regularizers)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        feature_extraction_layer = keras.saving.deserialize_keras_object(config.pop("feature_extraction_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        inference_layer = keras.saving.deserialize_keras_object(config.pop("inference_layer"))
        attribute_proc_layer = keras.saving.deserialize_keras_object(config.pop("attribute_processing_layer"))
        attribute_regularizers = keras.saving.deserialize_keras_object(config.pop("attribute_regularizers"))
        return cls(
            feature_extraction_layer=feature_extraction_layer,
            generative_layer=generative_layer,
            inference_layer=inference_layer,
            attribute_processing_layer=attribute_proc_layer,
            attribute_regularizers=attribute_regularizers,
            **config
        )
