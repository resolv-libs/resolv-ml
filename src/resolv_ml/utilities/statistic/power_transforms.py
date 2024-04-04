# TODO - DOC
from abc import ABC, abstractmethod
from typing import Tuple

import keras
import keras.ops as k_ops


class PowerTransform(ABC, keras.Layer):

    def __init__(self,
                 lambda_init: float = 1.0,
                 batch_norm: keras.layers.BatchNormalization = None,
                 name: str = "power_transform", **kwargs):
        super(PowerTransform, self).__init__(name=name, **kwargs)
        self._lambda_init = lambda_init
        self._batch_normalization = batch_norm

    @abstractmethod
    def _transform(self, inputs):
        pass

    @abstractmethod
    def _inverse_transform(self, inputs):
        pass

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        self.lmbda = self.add_weight(
            initializer=keras.initializers.Constant(self._lambda_init),
            trainable=True,
            name="lambda"
        )

    def call(self, inputs, inverse: bool = False, training: bool = False, **kwargs):
        transformed = self._transform(inputs) if not inverse else self._inverse_transform(inputs)
        return transformed if not self._batch_normalization or inverse else self._batch_normalization(transformed)


class BoxCox(PowerTransform):

    def __init__(self,
                 lambda_init: float = 1.0,
                 batch_norm: keras.layers.BatchNormalization = None,
                 name: str = "box_cox_power_transform", **kwargs):
        super(BoxCox, self).__init__(lambda_init=lambda_init, batch_norm=batch_norm, name=name, **kwargs)

    def _transform(self, inputs):
        # WARNING: input values must be > 0.
        # WARNING: when lambda = 0, Box-Cox returns log(x), the output ceases to depend on lambda. Therefore, the
        # gradient  w.r.t. lambda becomes identically zero. As such, lambda is not connected to the autograd graph
        # anymore, and cannot be optimized via (simple) gradient descent. However, it might still work with inertial
        # optimizers, i.e., those that have accumulated previous gradients, e.g., SGD with momentum, Adam, etc.
        return k_ops.cond(pred=k_ops.abs(self.lmbda) == 0,
                          true_fn=lambda: k_ops.log(inputs),
                          false_fn=lambda: (k_ops.power(inputs, self.lmbda) - 1) / self.lmbda)

    def _inverse_transform(self, inputs):
        return k_ops.cond(pred=k_ops.abs(self.lmbda) == 0,
                          true_fn=lambda: k_ops.exp(inputs),
                          false_fn=lambda: k_ops.power(inputs * self.lmbda + 1, 1 / self.lmbda))


class YeoJohnson(PowerTransform):

    def __init__(self,
                 lambda_init: float = 1.0,
                 batch_norm: keras.layers.BatchNormalization = None,
                 name: str = "yeo_johnson_power_transform", **kwargs):
        super(YeoJohnson, self).__init__(lambda_init=lambda_init, batch_norm=batch_norm, name=name, **kwargs)

    def _transform(self, inputs):
        x_pos, x_neg = self._get_positive_and_negative_inputs(inputs)
        y_pos = k_ops.cond(pred=k_ops.abs(self.lmbda) == 0,
                           true_fn=lambda: k_ops.log1p(x_pos),
                           false_fn=lambda: (k_ops.power(x_pos + 1, self.lmbda) - 1) / self.lmbda)

        y_neg = k_ops.cond(pred=k_ops.abs(self.lmbda) == 2,
                           true_fn=lambda: -k_ops.log1p(-x_neg),
                           false_fn=lambda: -(k_ops.power(-x_neg + 1, 2 - self.lmbda) - 1) / (2 - self.lmbda))
        return y_pos + y_neg

    def _inverse_transform(self, inputs):
        x_pos, x_neg = self._get_positive_and_negative_inputs(inputs)
        y_pos = k_ops.cond(pred=k_ops.abs(self.lmbda) == 0,
                           true_fn=lambda: k_ops.exp(x_pos) - 1,
                           false_fn=lambda: k_ops.power(self.lmbda * x_pos + 1, 1 / self.lmbda) - 1)

        y_neg = k_ops.cond(pred=k_ops.abs(self.lmbda) == 2,
                           true_fn=lambda: 1 - k_ops.exp(-x_neg),
                           false_fn=lambda: 1 - k_ops.power(-(2 - self.lmbda) * x_neg + 1, 1 / (2 - self.lmbda)))
        return y_pos + y_neg

    @staticmethod
    def _get_positive_and_negative_inputs(tensor):
        mask = k_ops.cast(k_ops.greater_equal(tensor, 0), dtype='float32')
        x_pos = k_ops.multiply(tensor, mask)
        x_neg = k_ops.multiply(tensor, k_ops.subtract(1, mask))
        return x_pos, x_neg
