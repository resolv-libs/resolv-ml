# TODO - DOC
import keras

from resolv_ml.models.nn.base import ResidualBlock
from resolv_ml.models.nn.embeddings import SinusoidalPositionalEncoding
from resolv_ml.models.nn.film import FiLM, FeatureWiseAffine


@keras.saving.register_keras_serializable(package="Denoisers", name="DenseDenoiser")
class DenseDenoiser(keras.Layer):
    """
    Represents a dense denoising neural network layer designed for processing embedded representations and
    applying conditioning through feature-wise transformations.

    This class outlines the construction of a configurable denoising layer with positional encoding
    and FiLM conditioning. It includes residual connections capable of processing features through multiple
    dense projection layers, applying learned transformations to inputs.

    :ivar positional_encoding_layer: Layer used for applying sinusoidal or other positional encodings
        to the input.
    :type positional_encoding_layer: keras.Layer
    :ivar units: Number of units in each dense layer within the denoising block.
    :type units: int
    :ivar num_layers: Number of residual layers in the denoiser.
    :type num_layers: int
    """

    @keras.saving.register_keras_serializable(package="Denoisers", name="DenseProjectionBlock")
    class FilMDenseProjectionBlock(keras.Layer):

        def __init__(self,
                     units: int = 2048,
                     num_layers: int = 2,
                     name="dense_proj_block",
                     **kwargs):
            super(DenseDenoiser.FilMDenseProjectionBlock, self).__init__(name=name, **kwargs)
            self.units = units
            self.num_layers = num_layers

        def build(self, input_shape):
            self._norm_layers = [keras.layers.LayerNormalization() for _ in range(self.num_layers)]
            self._fw_affine_layers = [FeatureWiseAffine(activation="silu") for _ in range(self.num_layers)]
            self._dense_layers = [keras.layers.Dense(self.units) for _ in range(self.num_layers)]

        def call(self, inputs, scale, shift, training: bool = False):
            x = inputs
            output = x
            for i in range(self.num_layers):
                output = self._norm_layers[i](output)
                output = self._fw_affine_layers[i]((output, scale, shift))
                output = self._dense_layers[i](output)
            return output

        def get_config(self):
            base_config = super().get_config()
            config = {
                "units": self.units,
                "num_layers": self.num_layers
            }
            return {**base_config, **config}

    def __init__(self,
                 units: int = 2048,
                 num_layers: int = 3,
                 positional_encoding_layer: keras.Layer = SinusoidalPositionalEncoding(embedding_dim=128),
                 name="denoiser",
                 **kwargs):
        super(DenseDenoiser, self).__init__(name=name, **kwargs)
        self.positional_encoding_layer = positional_encoding_layer
        self.units = units
        self.num_layers = num_layers

    def build(self, input_shape):
        super().build(input_shape)
        x_shape, conditioning_shape = input_shape
        _, embedding_channels = self.positional_encoding_layer.compute_output_shape(conditioning_shape)
        dense_units = embedding_channels * 4
        self._cond_layers = [
            self.positional_encoding_layer,
            keras.layers.Dense(dense_units, activation='silu'),
            keras.layers.Dense(dense_units),
            FiLM(gamma_layer=keras.layers.Dense(self.units), beta_layer=keras.layers.Dense(self.units))
        ]
        self._input_layers = [keras.layers.Dense(self.units)]
        self._dense_res_block_layers = [
            ResidualBlock(
                projection_connection_fn=DenseDenoiser.FilMDenseProjectionBlock(units=self.units),
                residual_fn=keras.layers.Dense(self.units) if x_shape[-1] != self.units else None,
            ) for _ in range(self.num_layers)
        ]
        self._output_layers = [keras.layers.LayerNormalization(), keras.layers.Dense(x_shape[-1])]
        # Call the layer with placeholder inputs to build it
        self.call(inputs=[keras.Input(shape=x_shape), None, keras.Input(shape=(x_shape[0], 1))])

    def call(self, inputs, training: bool = False):
        x, _, conditioning = inputs
        scale, shift = self._get_film_from_conditioning(conditioning)
        for layer in self._input_layers:
            x = layer(x, training=training)
        for layer in self._dense_res_block_layers:
            x = layer(x, training=training, scale=scale, shift=shift)
        for layer in self._output_layers:
            x = layer(x, training=training)
        return x

    def _get_film_from_conditioning(self, conditioning):
        output = conditioning
        for layer in self._cond_layers:
            output = layer(output)
        return output

    def get_config(self):
        base_config = super().get_config()
        config = {
            "positional_encoding_layer": keras.saving.serialize_keras_object(self.positional_encoding_layer),
            "units": self.units,
            "num_layers": self.num_layers
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        positional_encoding_layer = keras.saving.deserialize_keras_object(config.pop("positional_encoding_layer"))
        return cls(
            positional_encoding_layer=positional_encoding_layer,
            **config
        )
