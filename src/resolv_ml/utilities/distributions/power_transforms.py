# TODO - DOC
import math
from abc import ABC
from typing import Tuple

import keras
from keras import ops as k_ops


class PowerTransform(keras.Layer):

    def __init__(self,
                 power_initializer: keras.Initializer = keras.initializers.Constant(0.0),
                 trainable: bool = True,
                 batch_norm: keras.layers.BatchNormalization = None,
                 name: str = "power_transform", **kwargs):
        super(PowerTransform, self).__init__(name=name, **kwargs)
        self._power_initializer = power_initializer
        self._trainable = trainable
        self._batch_normalization = batch_norm

    def transform(self, inputs):
        raise NotImplementedError("Subclasses of `PowerTransform` must implement the `transform()` method.")

    def inverse_transform(self, inputs):
        raise NotImplementedError("Subclasses of `PowerTransform` must implement the `inverse_transform()` method.")

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape: Tuple[int, ...]):
        self.power = self.add_weight(
            initializer=self._power_initializer,
            trainable=self._trainable,
            name="power"
        )

    def call(self, inputs, inverse: bool = False, training: bool = False, **kwargs):
        if inverse:
            inverse_transform = self.inverse_transform(inputs)
            return self.inverse_batch_norm(inverse_transform) if self._batch_normalization else inverse_transform
        else:
            transformed = self.transform(inputs)
            return self._batch_normalization(transformed, training=training) if self._batch_normalization else transformed

    def inverse_batch_norm(self, inputs):
        mean = self._batch_normalization.moving_mean
        variance = self._batch_normalization.moving_variance
        epsilon = self._batch_normalization.epsilon
        beta = self._batch_normalization.beta
        gamma = self._batch_normalization.gamma
        return ((inputs - beta) / gamma) * keras.ops.sqrt(variance + epsilon) + mean

    def get_config(self):
        base_config = super().get_config()
        config = {
            "trainable": self._trainable,
            "power_initializer": keras.saving.serialize_keras_object(self._power_initializer),
            "batch_norm": keras.saving.serialize_keras_object(self._batch_normalization)
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="PowerTransform", name="BoxCox")
class BoxCox(PowerTransform):

    def __init__(self,
                 power_initializer: keras.Initializer = keras.initializers.Constant(0.0),
                 shift_initializer: keras.Initializer = keras.initializers.Constant(0.0),
                 trainable: bool = True,
                 batch_norm: keras.layers.BatchNormalization = None,
                 name: str = "box_cox", **kwargs):
        super(BoxCox, self).__init__(power_initializer=power_initializer,
                                     trainable=trainable,
                                     batch_norm=batch_norm,
                                     name=name,
                                     **kwargs)
        self._shift_initializer = shift_initializer

    def build(self, input_shape: Tuple[int, ...]):
        super().build(input_shape)
        self.shift = self.add_weight(
            initializer=self._shift_initializer,
            trainable=self._trainable,
            name="shift"
        )

    def transform(self, inputs):
        return k_ops.cond(pred=k_ops.abs(self.power) == 0,
                          true_fn=lambda: k_ops.log(inputs + self.shift),
                          false_fn=lambda: ((inputs + self.shift) ** self.power - 1) / self.power)

    def inverse_transform(self, inputs):
        return k_ops.cond(pred=k_ops.abs(self.power) == 0,
                          true_fn=lambda: k_ops.exp(inputs) - self.shift,
                          false_fn=lambda: (inputs * self.power + 1) ** (1 / self.power) - self.shift)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "shift_initializer": keras.saving.serialize_keras_object(self._shift_initializer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        power_initializer = keras.saving.serialize_keras_object(config.pop("power_initializer"))
        shift_initializer = keras.saving.serialize_keras_object(config.pop("shift_initializer"))
        batch_norm = keras.saving.deserialize_keras_object(config.pop("batch_norm"))
        return cls(power_initializer=power_initializer,
                   shift_initializer=shift_initializer,
                   batch_norm=batch_norm,
                   **config)


@keras.saving.register_keras_serializable(package="PowerTransform", name="YeoJohnson")
class YeoJohnson(PowerTransform):

    def __init__(self,
                 power_initializer: keras.Initializer = keras.initializers.Constant(0.0),
                 trainable: bool = True,
                 batch_norm: keras.layers.BatchNormalization = None,
                 name: str = "yeo_johnson", **kwargs):
        super(YeoJohnson, self).__init__(power_initializer=power_initializer,
                                         trainable=trainable,
                                         batch_norm=batch_norm,
                                         name=name,
                                         **kwargs)

    def transform(self, inputs):
        x_pos, x_neg = self._get_positive_and_negative_inputs(inputs)
        y_pos = k_ops.cond(pred=k_ops.abs(self.power) == 0,
                           true_fn=lambda: k_ops.log1p(x_pos),
                           false_fn=lambda: (k_ops.power(x_pos + 1, self.power) - 1) / self.power)

        y_neg = k_ops.cond(pred=k_ops.abs(self.power) == 2,
                           true_fn=lambda: -k_ops.log1p(-x_neg),
                           false_fn=lambda: -(k_ops.power(-x_neg + 1, 2 - self.power) - 1) / (2 - self.power))
        return y_pos + y_neg

    def inverse_transform(self, inputs):
        x_pos, x_neg = self._get_positive_and_negative_inputs(inputs)
        y_pos = k_ops.cond(pred=k_ops.abs(self.power) == 0,
                           true_fn=lambda: k_ops.exp(x_pos) - 1,
                           false_fn=lambda: k_ops.power(self.power * x_pos + 1, 1 / self.power) - 1)

        y_neg = k_ops.cond(pred=k_ops.abs(self.power) == 2,
                           true_fn=lambda: 1 - k_ops.exp(-x_neg),
                           false_fn=lambda: 1 - k_ops.power(-(2 - self.power) * x_neg + 1, 1 / (2 - self.power)))
        return y_pos + y_neg

    @staticmethod
    def _get_positive_and_negative_inputs(tensor):
        mask = k_ops.cast(k_ops.greater_equal(tensor, 0), dtype='float32')
        x_pos = k_ops.multiply(tensor, mask)
        x_neg = k_ops.multiply(tensor, k_ops.subtract(1, mask))
        return x_pos, x_neg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        power_initializer = keras.saving.serialize_keras_object(config.pop("power_initializer"))
        batch_norm = keras.saving.deserialize_keras_object(config.pop("batch_norm"))
        return cls(power_initializer=power_initializer,
                   batch_norm=batch_norm,
                   **config)
