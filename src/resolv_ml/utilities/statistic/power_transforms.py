# TODO - DOC
from abc import ABC, abstractmethod
from typing import Tuple

import keras
import keras.ops as k_ops


class PowerTransform(ABC, keras.Layer):

    def __init__(self, lambda_init: float = 1.0, name: str = "power_transform", **kwargs):
        super(PowerTransform, self).__init__(name=name, **kwargs)
        self.lambda_init = lambda_init

    @abstractmethod
    def _transform(self, inputs):
        pass

    @abstractmethod
    def _inverse_transform(self, inputs):
        pass

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        self.lmbda = self.add_weight(
            initializer=keras.initializers.Constant(self.lambda_init),
            trainable=True,
            name="lambda"
        )

    def call(self, inputs, inverse: bool = False, training: bool = False, **kwargs):
        return self._transform(inputs) if not inverse else self._inverse_transform(inputs)


class BoxCox(PowerTransform):

    def __init__(self, lambda_init: float = 1.0, name: str = "box_cox_power_transform", **kwargs):
        super(BoxCox, self).__init__(lambda_init=lambda_init, name=name, **kwargs)

    def _transform(self, inputs):
        def error_fn():
            # TODO - Support for graph execution
            raise ValueError('The Box-Cox transformation can only be applied to strictly positive data.')

        def zero_lambda_fn():
            # WARNING: when Box-Cox returns log(x), the output ceases to depend on lambda. Therefore, the gradient
            # w.r.t. lambda becomes identically zero. As such, lambda is not connected to the autograd graph anymore,
            # and cannot be optimized via (simple) gradient descent. However, it might still work with inertial
            # optimizers, i.e., those that have accumulated previous gradients, e.g., SGD with momentum, Adam, etc.
            return k_ops.log(inputs)

        def non_zero_lambda_fn():
            numerator = k_ops.subtract(k_ops.power(inputs, self.lmbda), 1)
            denominator = self.lmbda
            return k_ops.divide(numerator, denominator)

        k_ops.cond(k_ops.any(k_ops.less_equal(inputs, 0)), true_fn=error_fn, false_fn=lambda: ())
        return k_ops.cond(k_ops.equal(k_ops.abs(self.lmbda), 0), zero_lambda_fn, non_zero_lambda_fn)

    def _inverse_transform(self, inputs):
        def zero_lambda_fn():
            return k_ops.exp(inputs)

        def non_zero_lambda_fn():
            power_base = k_ops.add(k_ops.multiply(inputs, self.lmbda), 1)
            power_exp = k_ops.divide(1, self.lmbda)
            return k_ops.power(power_base, power_exp)

        return k_ops.cond(k_ops.equal(k_ops.abs(self.lmbda), 0), zero_lambda_fn, non_zero_lambda_fn)


class YeoJohnson(PowerTransform):

    def __init__(self, lambda_init: float = 1.0, name: str = "yeo_johnson_power_transform", **kwargs):
        super(YeoJohnson, self).__init__(lambda_init=lambda_init, name=name, **kwargs)

    def _transform(self, inputs):
        def zero_lambda_fn():
            return k_ops.log1p(x_pos)

        def non_zero_lambda_fn():
            numerator = k_ops.subtract(k_ops.power(k_ops.add(x_pos, 1), self.lmbda), 1)
            denominator = self.lmbda
            return k_ops.divide(numerator, denominator)

        def two_lambda_fn():
            return -k_ops.log1p(-x_neg)

        def non_two_lambda_fn():
            numerator = -(k_ops.power(k_ops.add(-x_neg, 1), k_ops.subtract(2, self.lmbda) - 1))
            denominator = k_ops.subtract(2, self.lmbda)
            return k_ops.divide(numerator, denominator)

        x_pos, x_neg = self._get_positive_and_negative_inputs(inputs)
        y_pos = k_ops.cond(k_ops.equal(k_ops.abs(self.lmbda), 0), zero_lambda_fn, non_zero_lambda_fn)
        y_neg = k_ops.cond(k_ops.equal(k_ops.abs(self.lmbda), 2), two_lambda_fn, non_two_lambda_fn)
        return y_pos + y_neg

    def _inverse_transform(self, inputs):
        def zero_lambda_fn():
            return k_ops.subtract(k_ops.exp(x_pos), 1)

        def non_zero_lambda_fn():
            power_base = k_ops.add(k_ops.multiply(self.lmbda, x_pos), 1)
            power_exp = k_ops.divide(1, self.lmbda)
            return k_ops.subtract(k_ops.power(power_base, power_exp), 1)

        def two_lambda_fn():
            return k_ops.subtract(1, k_ops.exp(-x_neg))

        def non_two_lambda_fn():
            power_base = k_ops.add(k_ops.multiply(-k_ops.subtract(2, self.lmbda), x_neg), 1)
            power_exp = k_ops.divide(1, k_ops.subtract(2, self.lmbda))
            return k_ops.subtract(1, k_ops.power(power_base, power_exp))

        x_pos, x_neg = self._get_positive_and_negative_inputs(inputs)
        y_pos = k_ops.cond(k_ops.equal(k_ops.abs(self.lmbda), 0), zero_lambda_fn, non_zero_lambda_fn)
        y_neg = k_ops.cond(k_ops.equal(k_ops.abs(self.lmbda), 2), two_lambda_fn, non_two_lambda_fn)
        return y_pos + y_neg

    @staticmethod
    def _get_positive_and_negative_inputs(tensor):
        mask = k_ops.cast(k_ops.greater_equal(tensor, 0), dtype='float32')
        x_pos = k_ops.multiply(tensor, mask)
        x_neg = k_ops.multiply(tensor, k_ops.subtract(1, mask))
        return x_pos, x_neg
