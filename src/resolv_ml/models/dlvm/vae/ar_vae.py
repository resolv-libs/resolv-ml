# TODO - DOC
import keras

from .vanilla_vae import StandardVAE
from ....utilities.regularizers.attribute import AttributeRegularizer
from ....utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="VAE", name="AttributeRegularizedVAE")
class AttributeRegularizedVAE(StandardVAE):

    def __init__(self,
                 z_size: int,
                 input_processing_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 attribute_processing_layer: keras.Layer,
                 attribute_regularization_layer: AttributeRegularizer,
                 mean_inference_layer: keras.Layer = None,
                 sigma_inference_layer: keras.Layer = None,
                 div_beta_scheduler: Scheduler = None,
                 free_bits: float = 0.0,
                 name: str = "ar_vae",
                 **kwargs):
        self._attribute_processing_layer = attribute_processing_layer
        self._attribute_regularization_layer = attribute_regularization_layer
        self._attr_loss_tracker = keras.metrics.Mean(name=attribute_regularization_layer.regularization_name)
        self._attr_gamma_tracker = keras.metrics.Mean(
            name=f"{attribute_regularization_layer.regularization_name}_gamma"
        )
        self._attr_loss_weighted_tracker = keras.metrics.Mean(
            name=f"{attribute_regularization_layer.regularization_name}_weighted"
        )
        super(AttributeRegularizedVAE, self).__init__(
            z_size=z_size,
            input_processing_layer=input_processing_layer,
            aux_input_processing_layer=attribute_processing_layer,
            generative_layer=generative_layer,
            mean_inference_layer=mean_inference_layer,
            sigma_inference_layer=sigma_inference_layer,
            div_beta_scheduler=div_beta_scheduler,
            free_bits=free_bits,
            name=name,
            **kwargs
        )
        self._regularization_layers["attr_reg"] = attribute_regularization_layer

    def _add_regularization_losses(self, regularization_losses):
        super()._add_regularization_losses(regularization_losses)
        attr_reg_loss, attr_reg_gamma = regularization_losses[1]
        weighted_reg_loss = attr_reg_gamma * attr_reg_loss
        self.add_loss(weighted_reg_loss)
        self._attr_loss_tracker.update_state(attr_reg_loss)
        self._attr_gamma_tracker.update_state(attr_reg_gamma)
        self._attr_loss_weighted_tracker.update_state(weighted_reg_loss)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "attribute_processing_layer": keras.saving.serialize_keras_object(self._attribute_processing_layer),
            "attribute_regularization_layer": keras.saving.serialize_keras_object(self._attribute_regularization_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_processing_layer = keras.saving.deserialize_keras_object(config.pop("input_processing_layer"))
        mean_inference_layer = keras.saving.deserialize_keras_object(config.pop("mean_inference_layer"))
        sigma_inference_layer = keras.saving.deserialize_keras_object(config.pop("sigma_inference_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        attribute_proc_layer = keras.saving.deserialize_keras_object(config.pop("attribute_processing_layer"))
        attribute_reg_layer = keras.saving.deserialize_keras_object(config.pop("attribute_regularization_layer"))
        return cls(
            input_processing_layer=input_processing_layer,
            mean_inference_layer=mean_inference_layer,
            sigma_inference_layer=sigma_inference_layer,
            generative_layer=generative_layer,
            attribute_processing_layer=attribute_proc_layer,
            attribute_regularization_layer=attribute_reg_layer,
            **config
        )
