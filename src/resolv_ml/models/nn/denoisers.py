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

    :ivar timestep_encoding_layer: Layer used for applying sinusoidal or other positional encodings to the timestep.
    :type timestep_encoding_layer: keras.Layer
    :ivar labels_encoding_layer: Layer used for applying sinusoidal or other positional encodings to the labels.
    :type labels_encoding_layer: keras.Layer
    :ivar units: Number of units in each dense layer within the denoising block.
    :type units: int
    :ivar num_layers: Number of residual layers in the denoiser.
    :type num_layers: int
    :ivar conditioning_merge_mode: The mode in which conditioning signals are merged. Applied only if the denoiser in
     conditioned by more than one signal. Accepted values: sum, prod and concat.
    :type conditioning_merge_mode: str
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
                 timestep_encoding_layer: keras.Layer = SinusoidalPositionalEncoding(embedding_dim=128),
                 labels_encoding_layer: keras.Layer = SinusoidalPositionalEncoding(embedding_dim=128),
                 conditioning_merge_mode: str = 'sum',
                 name="denoiser",
                 **kwargs):
        super(DenseDenoiser, self).__init__(name=name, **kwargs)
        if conditioning_merge_mode not in ['sum', 'prod', 'concat']:
            raise ValueError("Invalid conditioning merge mode.")
        self.timestep_encoding_layer = timestep_encoding_layer
        self.labels_encoding_layer = labels_encoding_layer
        self.units = units
        self.num_layers = num_layers
        self.conditioning_merge_mode = conditioning_merge_mode

    def build(self, input_shape):
        super().build(input_shape)
        x_shape, conditioning_shape = input_shape
        _, timestep_emb_channels = self.timestep_encoding_layer.compute_output_shape(conditioning_shape)
        timestep_dense_units = timestep_emb_channels * 4
        self._timestep_cond_layers = [
            self.timestep_encoding_layer,
            keras.layers.Dense(timestep_dense_units, activation='silu'),
            FiLM(gamma_layer=keras.layers.Dense(self.units), beta_layer=keras.layers.Dense(self.units))
        ]
        _, labels_emb_channels = self.labels_encoding_layer.compute_output_shape(conditioning_shape)
        labels_dense_units = labels_emb_channels * 4
        self._labels_cond_layers = [
            self.labels_encoding_layer,
            keras.layers.Dense(labels_dense_units, activation='silu'),
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
        self.call(
            inputs=[keras.Input(shape=x_shape[1:]), keras.Input(shape=(1,)), keras.Input(shape=(1,))]
        )

    def call(self, inputs, training: bool = False):
        x, x_labels, timestep_cond = inputs
        scale, shift = self._get_film_from_conditioning(x_labels, timestep_cond)
        for layer in self._input_layers:
            x = layer(x, training=training)
        for layer in self._dense_res_block_layers:
            x = layer(x, training=training, scale=scale, shift=shift)
        for layer in self._output_layers:
            x = layer(x, training=training)
        return x

    def _get_film_from_conditioning(self, labels, timestep_cond):
        # Timestep
        output = timestep_cond
        for layer in self._timestep_cond_layers:
            output = layer(output)
        timestep_scale, timestep_shift = output
        # Labels
        if labels is not None:
            output = labels
            for layer in self._labels_cond_layers:
                output = layer(output)
            labels_scale, labels_shift = output
        # Merge
        match self.conditioning_merge_mode:
            case "sum":
                if labels is None:
                    labels_scale, labels_shift = (keras.ops.zeros_like(timestep_scale),
                                                  keras.ops.zeros_like(timestep_shift))
                return (keras.ops.add(timestep_scale, labels_scale),
                        keras.ops.add(timestep_shift, labels_shift))
            case "prod":
                if labels is None:
                    labels_scale, labels_shift = (keras.ops.ones_like(timestep_scale),
                                                  keras.ops.ones_like(timestep_shift))
                return (keras.ops.multiply(timestep_scale, labels_scale),
                        keras.ops.multiply(timestep_shift, labels_shift))
            case _:
                if labels is None:
                    # TODO - What is the neutral element for concatenation?
                    labels_scale, labels_shift = (keras.ops.zeros_like(timestep_scale),
                                                  keras.ops.zeros_like(timestep_shift))
                return (keras.ops.concatenate([timestep_scale, labels_scale], axis=-1),
                        keras.ops.concatenate([timestep_shift, labels_shift], axis=-1))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "timestep_encoding_layer": keras.saving.serialize_keras_object(self.timestep_encoding_layer),
            "labels_encoding_layer": keras.saving.serialize_keras_object(self.labels_encoding_layer),
            "units": self.units,
            "num_layers": self.num_layers,
            "conditioning_merge_mode": self.conditioning_merge_mode
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        timestep_encoding_layer = keras.saving.deserialize_keras_object(config.pop("timestep_encoding_layer"))
        labels_encoding_layer = keras.saving.deserialize_keras_object(config.pop("labels_encoding_layer"))
        return cls(
            timestep_encoding_layer=timestep_encoding_layer,
            labels_encoding_layer=labels_encoding_layer,
            **config
        )
