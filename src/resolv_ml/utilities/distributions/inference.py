# TODO - DOC
from typing import Any

import keras
from keras import ops as k_ops


@keras.saving.register_keras_serializable(package="Inference", name="GaussianInference")
class GaussianInference(keras.Layer):

    def __init__(self,
                 z_size: int,
                 mean_layer: keras.Layer = None,
                 log_var_layer: keras.Layer = None,
                 name: str = "gaussian_inference",
                 **kwargs):
        super(GaussianInference, self).__init__(name=name, **kwargs)
        self._z_size = z_size
        self._mean_layer = mean_layer if mean_layer else self.default_mean_layer(z_size)
        self._log_var_layer = log_var_layer if log_var_layer else self.default_log_var_layer(z_size)

    def build(self, input_shape):
        if not self._mean_layer.built:
            self._mean_layer.build(input_shape)
        if not self._log_var_layer.built:
            self._log_var_layer.build(input_shape)

    def call(self, inputs: Any, training: bool = False, **kwargs):
        z_mean = self._mean_layer(inputs)
        z_log_var = self._log_var_layer(inputs)
        return z_mean, k_ops.exp(z_log_var)

    def compute_output_shape(self, input_shape):
        return (self._mean_layer.compute_output_shape(input_shape),
                self._log_var_layer.compute_output_shape(input_shape))

    @classmethod
    def default_mean_layer(cls, z_size: int) -> keras.layers.Dense:
        return keras.layers.Dense(
            units=z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name="z_mean"
        )

    @classmethod
    def default_log_var_layer(cls, z_size: int) -> keras.layers.Dense:
        return keras.layers.Dense(
            units=z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            bias_initializer="zeros",
            name="z_log_var"
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_size": self._z_size,
            "mean_layer": keras.saving.serialize_keras_object(self._mean_layer),
            "log_var_layer": keras.saving.serialize_keras_object(self._log_var_layer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        mean_layer = keras.saving.deserialize_keras_object(config.pop("mean_layer"))
        log_var_layer = keras.saving.deserialize_keras_object(config.pop("log_var_layer"))
        return cls(mean_layer=mean_layer, log_var_layer=log_var_layer, **config)


@keras.saving.register_keras_serializable(package="Inference", name="GaussianReparametrizationTrick")
class GaussianReparametrizationTrick(keras.Layer):

    def __init__(self, z_size: int, name: str = "reparametrization_trick", **kwargs):
        super(GaussianReparametrizationTrick, self).__init__(name=name, **kwargs)
        self._z_size = z_size

    def compute_output_shape(self, input_shape, **kwargs):
        return input_shape[0]

    def call(self, inputs, training: bool = False, **kwargs):
        z_mean, z_var, _ = inputs
        epsilon = keras.random.normal(shape=k_ops.shape(z_mean), mean=0.0, stddev=1.0)
        z = z_mean + k_ops.sqrt(z_var) * epsilon
        return z
