# TODO - DOC

import keras
from keras.src import activations

from ...utilities.utils import filter_kwargs


@keras.saving.register_keras_serializable(package="NeuralNetworks", name="ResidualBlock")
class ResidualBlock(keras.Layer):

    def __init__(self,
                 residual_fn: keras.Layer = None,
                 projection_connection_fn: keras.Layer = None,
                 pre_activation_fn: str = None,
                 post_activation_fn: str = None,
                 name: str = "residual_block",
                 **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        self._residual_fn = residual_fn if residual_fn else keras.layers.Identity()
        self._projection_connection_fn = projection_connection_fn if projection_connection_fn \
            else keras.layers.Identity()
        self._pre_activation_fn = pre_activation_fn
        self._pre_activation = activations.get(self._pre_activation_fn)
        self._post_activation_fn = post_activation_fn
        self._post_activation = activations.get(self._post_activation_fn)

    def compute_output_shape(self, input_shape):
        return self._residual_fn.compute_output_shape(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        # TODO - keras CallSpec bug?
        kwargs = kwargs['kwargs'] if 'kwargs' in kwargs else kwargs
        if self._pre_activation:
            inputs = self._pre_activation(inputs)
        residual = self._residual_fn(inputs, training=training, **filter_kwargs(self._residual_fn.call, **kwargs))
        projection = self._projection_connection_fn(
            inputs, training=training, **filter_kwargs(self._projection_connection_fn.call, **kwargs)
        )
        output = residual + projection
        if self._post_activation:
            output = self._post_activation(output)
        return output

    def get_config(self):
        base_config = super().get_config()
        config = {
            "residual_fn": keras.saving.serialize_keras_object(self._residual_fn),
            "projection_connection_fn": keras.saving.serialize_keras_object(self._projection_connection_fn),
            "pre_activation_fn": self._pre_activation_fn,
            "post_activation_fn": self._post_activation_fn
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        residual_fn = keras.saving.deserialize_keras_object(config.pop("residual_fn"))
        projection_connection_fn = keras.saving.deserialize_keras_object(config.pop("projection_connection_fn"))
        pre_activation_fn = keras.saving.deserialize_keras_object(config.pop("pre_activation_fn"))
        post_activation_fn = keras.saving.deserialize_keras_object(config.pop("post_activation_fn"))
        return cls(
            residual_fn=residual_fn,
            projection_connection_fn=projection_connection_fn,
            pre_activation_fn=pre_activation_fn,
            post_activation_fn=post_activation_fn,
            **config
        )
