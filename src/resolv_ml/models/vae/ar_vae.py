# TODO - DOC
from abc import abstractmethod
from typing import Tuple

import keras
import keras.ops as k_ops

from .base import VAE, VAEEncoder, VAEDecoder, LatentCodeProcessingLayer
from resolv_ml.utilities.math.distances import compute_pairwise_distance_matrix
from resolv_ml.utilities.statistic.power_transforms import PowerTransform


class AttributeRegularizationLayer(LatentCodeProcessingLayer):

    def __init__(self,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 name: str = "vae/attr_reg",
                 batch_normalization: keras.layers.BatchNormalization = None,
                 **kwargs):
        super(AttributeRegularizationLayer, self).__init__(name=name, **kwargs)
        self._gamma = gamma
        self._regularization_dimension = regularization_dimension
        self._batch_normalization = batch_normalization

    @property
    @abstractmethod
    def regularization_name(self):
        pass

    @abstractmethod
    def _compute_attribute_regularization_loss(self, latent_codes, attributes, training: bool = False):
        pass

    @property
    def metrics(self):
        return [self.regularization_loss_tracker]

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        self.regularization_loss_tracker = keras.metrics.Mean(name=self.regularization_name)

    def _process_latent_code(self, latent_codes, model_inputs, training: bool = False, **kwargs):
        try:
            latent_dimension = latent_codes[:, self._regularization_dimension]
            reg_loss = self._compute_attribute_regularization_loss(latent_dimension, kwargs["attributes"], training)
            self.add_loss(self._gamma * reg_loss)
            self.regularization_loss_tracker.update_state(reg_loss)
            return latent_codes
        except KeyError as e:
            raise ValueError("VAE attribute regularization layer requires an 'attributes' item in kwargs.") from e


class DefaultAttributeRegularization(AttributeRegularizationLayer):

    def __init__(self,
                 loss_fn: keras.Loss = keras.losses.mean_absolute_error,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 batch_normalization: keras.layers.BatchNormalization = None,
                 name: str = "vae/default_attr_reg",
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


class SignAttributeRegularization(AttributeRegularizationLayer):

    def __init__(self,
                 loss_fn: keras.Loss = keras.losses.mean_absolute_error,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 scale_factor: float = 1.0,
                 name: str = "vae/sign_attr_reg",
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
        sign_loss = self._loss_fn(lc_tanh, attribute_sign)
        return self._gamma * sign_loss


class PowerTransformAttributeRegularization(DefaultAttributeRegularization):

    def __init__(self,
                 power_transform: PowerTransform,
                 loss_fn: keras.Loss = keras.losses.mean_absolute_error,
                 gamma: float = 1.0,
                 regularization_dimension: int = 0,
                 batch_normalization: keras.layers.BatchNormalization = None,
                 name: str = "vae/default_attr_reg",
                 **kwargs):
        super(PowerTransformAttributeRegularization, self).__init__(
            loss_fn=loss_fn,
            gamma=gamma,
            regularization_dimension=regularization_dimension,
            batch_normalization=batch_normalization,
            name=name,
            **kwargs
        )
        self._power_transform = power_transform

    @property
    def regularization_name(self):
        return "power_transform_attr_reg_loss"

    def _compute_attribute_regularization_loss(self, latent_codes, attributes, training: bool = False):
        attributes = self._power_transform(attributes, training=training)
        return super()._compute_attribute_regularization_loss(latent_codes, attributes)


class AttributeRegularizedVAE(keras.Model):

    def __init__(self,
                 encoder: VAEEncoder,
                 decoder: VAEDecoder,
                 attribute_regularization_layer: AttributeRegularizationLayer,
                 z_size: int,
                 input_layer: keras.Layer = None,
                 output_layer: keras.Layer = None,
                 free_bits: float = None,
                 max_beta: float = None,
                 beta_rate: float = None,
                 sampling_schedule: str = "constant",
                 sampling_rate: float = 0.0,
                 name: str = "vae",
                 **kwargs):
        super(AttributeRegularizedVAE, self).__init__(name=name, **kwargs)
        self._vae = VAE(
            encoder=encoder,
            decoder=decoder,
            z_size=z_size,
            input_layer=input_layer,
            output_layer=output_layer,
            z_processing_layer=attribute_regularization_layer,
            free_bits=free_bits,
            max_beta=max_beta,
            beta_rate=beta_rate,
            sampling_schedule=sampling_schedule,
            sampling_rate=sampling_rate,
            name=name
        )

    def call(self, inputs, training: bool = False, **kwargs):
        return self._vae(inputs, training=training, **kwargs)
