# TODO - DOC

import keras
import keras.ops as k_ops

from .vanilla_vae import StandardVAE
from ....utilities.math.distances import compute_pairwise_distance_matrix
from ....utilities.distributions.power_transforms import PowerTransform


class AttributeRegularizationLayer(keras.Layer):

    def __init__(self,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 name: str = "attr_reg",
                 batch_normalization: keras.layers.BatchNormalization = None,
                 **kwargs):
        super(AttributeRegularizationLayer, self).__init__(name=name, **kwargs)
        self._gamma = gamma
        self._regularization_dimension = regularization_dimension
        self._batch_normalization = batch_normalization

    @property
    def regularization_name(self):
        raise NotImplementedError("regularization_name property must be defined by subclasses.")

    def _compute_attribute_regularization_loss(self, latent_codes, attributes, training: bool = False):
        raise NotImplementedError("_compute_attribute_regularization_loss must be implemented by subclasses.")
    
    def build(self, input_shape):
        super().build(input_shape)
        if self._batch_normalization and not self._batch_normalization.built:
            self._batch_normalization.build(input_shape)
    
    def call(self, inputs, training: bool = False, **kwargs):
        _, attributes, _, z, _ = inputs
        latent_dimension = z[:, self._regularization_dimension]
        reg_loss = self._compute_attribute_regularization_loss(latent_dimension, attributes, training)
        return self._gamma * reg_loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "gamma": self._gamma,
            "regularization_dimension": self._regularization_dimension,
            "batch_normalization": keras.saving.serialize_keras_object(self._batch_normalization)
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="VAE", name="DefaultAttributeRegularization")
class DefaultAttributeRegularization(AttributeRegularizationLayer):

    def __init__(self,
                 loss_fn: keras.losses.Loss = keras.losses.mean_absolute_error,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 batch_normalization: keras.layers.BatchNormalization = None,
                 name: str = "default_attr_reg",
                 **kwargs):
        super(DefaultAttributeRegularization, self).__init__(
            gamma=gamma,
            regularization_dimension=regularization_dimension,
            batch_normalization=batch_normalization,
            name=name,
            **kwargs
        )
        self._loss_fn = loss_fn

    @property
    def regularization_name(self):
        return "default_attr_reg_loss"

    def _compute_attribute_regularization_loss(self, latent_codes, attributes, training: bool = False):
        if self._batch_normalization:
            attributes = self._batch_normalization(attributes, training=training)
        return self._loss_fn(latent_codes, attributes)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        batch_normalization = keras.saving.deserialize_keras_object(config.pop("batch_normalization"))
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(loss_fn=loss_fn, batch_normalization=batch_normalization, **config)


@keras.saving.register_keras_serializable(package="VAE", name="SignAttributeRegularization")
class SignAttributeRegularization(AttributeRegularizationLayer):

    def __init__(self,
                 loss_fn: keras.losses.Loss = keras.losses.mean_absolute_error,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 scale_factor: float = 1.0,
                 name: str = "sign_attr_reg",
                 **kwargs):
        super(SignAttributeRegularization, self).__init__(
            gamma=gamma,
            regularization_dimension=regularization_dimension,
            batch_normalization=None,
            name=name,
            **kwargs
        )
        self._loss_fn = loss_fn
        self._scale_factor = scale_factor

    @property
    def regularization_name(self):
        return "sign_attr_reg_loss"

    def _compute_attribute_regularization_loss(self, latent_codes, attributes, training: bool = False):
        lc_dist_mat = compute_pairwise_distance_matrix(latent_codes)
        attribute_dist_mat = compute_pairwise_distance_matrix(attributes)
        lc_tanh = k_ops.tanh(lc_dist_mat * self._scale_factor)
        attribute_sign = k_ops.sign(attribute_dist_mat)
        lc_tanh_reshaped = k_ops.reshape(lc_tanh, (-1,))
        attribute_sign_reshaped = k_ops.reshape(attribute_sign, (-1,))
        sign_loss = self._loss_fn(lc_tanh_reshaped, attribute_sign_reshaped)
        return sign_loss

    def get_config(self):
        base_config = super().get_config()
        base_config.pop("batch_normalization")
        config = {
            "scale_factor": self._scale_factor,
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(loss_fn=loss_fn, **config)


@keras.saving.register_keras_serializable(package="VAE", name="PowerTransformAttributeRegularization")
class PowerTransformAttributeRegularization(AttributeRegularizationLayer):

    def __init__(self,
                 power_transform: PowerTransform,
                 loss_fn: keras.losses.Loss = keras.losses.mean_absolute_error,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 name: str = "pt_attr_reg",
                 **kwargs):
        super(PowerTransformAttributeRegularization, self).__init__(
            gamma=gamma,
            regularization_dimension=regularization_dimension,
            batch_normalization=None,
            name=name,
            **kwargs
        )
        self._loss_fn = loss_fn
        self._power_transform = power_transform

    @property
    def regularization_name(self):
        return "power_transform_attr_reg_loss"

    def _compute_attribute_regularization_loss(self, latent_codes, attributes, training: bool = False):
        attributes = self._power_transform(attributes, training=training)
        return self._loss_fn(latent_codes, attributes)

    def build(self, input_shape):
        super().build(input_shape)
        if not self._power_transform.built:
            self._power_transform.build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config.pop("batch_normalization")
        config = {
            "power_transform": keras.saving.serialize_keras_object(self._power_transform),
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        power_transform = keras.saving.deserialize_keras_object(config.pop("power_transform"))
        return cls(loss_fn=loss_fn, power_transform=power_transform, **config)


@keras.saving.register_keras_serializable(package="VAE", name="AttributeRegularizedVAE")
class AttributeRegularizedVAE(StandardVAE):

    def __init__(self,
                 z_size: int,
                 input_processing_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 attribute_regularization_layer: AttributeRegularizationLayer = DefaultAttributeRegularization(),
                 mean_inference_layer: keras.Layer = None,
                 log_var_inference_layer: keras.Layer = None,
                 max_beta: float = 1.0,
                 beta_rate: float = 0.0,
                 free_bits: float = 0.0,
                 name: str = "ar_vae",
                 **kwargs):
        self._attribute_regularization_layer = attribute_regularization_layer
        self._attr_reg_loss_tracker = keras.metrics.Mean(name=attribute_regularization_layer.regularization_name)
        super(AttributeRegularizedVAE, self).__init__(
            z_size=z_size,
            input_processing_layer=input_processing_layer,
            generative_layer=generative_layer,
            mean_inference_layer=mean_inference_layer,
            log_var_inference_layer=log_var_inference_layer,
            free_bits=free_bits,
            max_beta=max_beta,
            beta_rate=beta_rate,
            name=name,
            **kwargs
        )
        self._regularization_layers.append(attribute_regularization_layer)

    @property
    def metrics(self):
        return super().metrics + [self._attr_reg_loss_tracker]

    def _add_regularization_losses(self, regularization_losses):
        super()._add_regularization_losses(regularization_losses)
        attr_reg_loss = regularization_losses[1]
        self.add_loss(attr_reg_loss)
        self._attr_reg_loss_tracker.update_state(attr_reg_loss)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "attribute_regularization_layer": keras.saving.serialize_keras_object(self._attribute_regularization_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_processing_layer = keras.saving.deserialize_keras_object(config.pop("input_processing_layer"))
        mean_inference_layer = keras.saving.deserialize_keras_object(config.pop("mean_inference_layer"))
        log_var_inference_layer = keras.saving.deserialize_keras_object(config.pop("log_var_inference_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        attribute_reg_layer = keras.saving.deserialize_keras_object(config.pop("attribute_regularization_layer"))
        return cls(
            input_processing_layer=input_processing_layer,
            mean_inference_layer=mean_inference_layer,
            log_var_inference_layer=log_var_inference_layer,
            generative_layer=generative_layer,
            attribute_regularization_layer=attribute_reg_layer,
            **config
        )
