# TODO - DOC
from typing import Tuple, Any

import keras
import keras.ops as k_ops


class GaussianInference(keras.Model):

    def __init__(self,
                 z_size: int,
                 mean_layer: keras.Layer = None,
                 log_var_layer: keras.Layer = None,
                 name: str = "gaussian_inference",
                 **kwargs):
        super(GaussianInference, self).__init__(name=name, **kwargs)
        self._z_size = z_size
        self._mean_layer = mean_layer
        self._log_var_layer = log_var_layer

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        self._mean_layer = self._mean_layer if self._mean_layer else self._default_mean_layer()
        self._log_var_layer = self._log_var_layer if self._log_var_layer else self._default_log_var_layer()

    def call(self, inputs: Any, training: bool = False, **kwargs):
        z_mean = self._mean_layer(inputs)
        z_log_var = self._log_var_layer(inputs)
        return z_mean, k_ops.exp(z_log_var)

    def _default_mean_layer(self):
        return keras.layers.Dense(
            units=self._z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name=f"{self.name}/z_mean"
        )

    def _default_log_var_layer(self):
        return keras.layers.Dense(
            units=self._z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            bias_initializer="zeros",
            name=f"{self.name}/z_log_var"
        )


class GaussianReparametrizationTrick(keras.Layer):

    def __init__(self, name: str = "reparametrization_trick", **kwargs):
        super(GaussianReparametrizationTrick, self).__init__(name=name, **kwargs)

    def call(self, z_mean, z_var, training: bool = False, **kwargs):
        epsilon = keras.random.normal(shape=k_ops.shape(z_mean), mean=0.0, stddev=1.0)
        z = z_mean + k_ops.sqrt(z_var) * epsilon
        return z
