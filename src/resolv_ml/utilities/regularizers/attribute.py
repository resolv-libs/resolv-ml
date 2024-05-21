# TODO - DOC
import keras
from keras import ops as k_ops
from tensorflow_probability import distributions as tfp_dist

from .base import Regularizer
from ..schedulers import Scheduler
from ...utilities.math.distances import compute_pairwise_distance_matrix
from ...utilities.distributions.power_transforms import PowerTransform


class AttributeRegularizer(Regularizer):

    def __init__(self,
                 beta_scheduler: Scheduler = None,
                 regularization_dimension: int = 0,
                 batch_normalization: keras.layers.BatchNormalization = None,
                 name: str = "attr_regularizer",
                 **kwargs):
        super(AttributeRegularizer, self).__init__(
            beta_scheduler=beta_scheduler,
            name=name,
            **kwargs
        )
        self._regularization_dimension = regularization_dimension
        self._batch_normalization = batch_normalization

    @property
    def regularization_name(self):
        raise NotImplementedError("regularization_name property must be defined by subclasses.")

    def _compute_attribute_regularization_loss(self, latent_codes, attributes, training: bool = False):
        raise NotImplementedError("_compute_attribute_regularization_loss must be implemented by subclasses.")

    def build(self, input_shape):
        vae_input_shape, aux_input_shape = input_shape
        super().build(aux_input_shape)
        if self._batch_normalization and not self._batch_normalization.built:
            self._batch_normalization.build(aux_input_shape)

    def _compute_regularization_loss(self,
                                     inputs,
                                     posterior: tfp_dist.Distribution,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        _, attributes, z, _ = inputs
        latent_dimension = k_ops.expand_dims(z[:, self._regularization_dimension], axis=-1)
        reg_loss = self._compute_attribute_regularization_loss(latent_dimension, attributes, training)
        return reg_loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "regularization_dimension": self._regularization_dimension,
            "batch_normalization": keras.saving.serialize_keras_object(self._batch_normalization)
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="Regularizers", name="DefaultAttributeRegularizer")
class DefaultAttributeRegularizer(AttributeRegularizer):

    def __init__(self,
                 beta_scheduler: Scheduler = None,
                 loss_fn: keras.losses.Loss = keras.losses.MeanAbsoluteError(),
                 regularization_dimension: int = 0,
                 batch_normalization: keras.layers.BatchNormalization = None,
                 name: str = "default_attr_regularizer",
                 **kwargs):
        super(DefaultAttributeRegularizer, self).__init__(
            beta_scheduler=beta_scheduler,
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


@keras.saving.register_keras_serializable(package="Regularizers", name="SignAttributeRegularizer")
class SignAttributeRegularizer(AttributeRegularizer):

    def __init__(self,
                 beta_scheduler: Scheduler = None,
                 loss_fn: keras.losses.Loss = keras.losses.MeanAbsoluteError(),
                 regularization_dimension: int = 0,
                 scale_factor: float = 1.0,
                 name: str = "sign_attr_regularizer",
                 **kwargs):
        super(SignAttributeRegularizer, self).__init__(
            beta_scheduler=beta_scheduler,
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


@keras.saving.register_keras_serializable(package="Regularizers", name="PowerTransformAttributeRegularizer")
class PowerTransformAttributeRegularizer(AttributeRegularizer):

    def __init__(self,
                 power_transform: PowerTransform,
                 beta_scheduler: Scheduler = None,
                 loss_fn: keras.losses.Loss = keras.losses.MeanAbsoluteError(),
                 regularization_dimension: int = 0,
                 name: str = "pt_attr_regularizer",
                 **kwargs):
        super(PowerTransformAttributeRegularizer, self).__init__(
            beta_scheduler=beta_scheduler,
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
