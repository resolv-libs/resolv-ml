# TODO - DOC
import keras

from ...utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="SequenceEncoders", name="SequenceEncoder")
class SequenceEncoder(keras.Model):

    def __init__(self,
                 embedding_layer: keras.Layer = None,
                 name: str = "seq_encoder",
                 **kwargs):
        super(SequenceEncoder, self).__init__(name=name, **kwargs)
        self._embedding_layer = embedding_layer

    def encode(self, inputs, training: bool = False, **kwargs):
        raise NotImplementedError("SequenceEncoder.encode must be overridden by subclasses.")

    def build(self, input_shape):
        super().build(input_shape)
        if self._embedding_layer:
            input_shape = self._check_embedding_layer_input_shape(input_shape)
            if not self._embedding_layer.built:
                self._embedding_layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        embedded_seq = inputs
        if self._embedding_layer:
            if isinstance(self._embedding_layer, keras.layers.Embedding):
                inputs = keras.ops.squeeze(inputs, axis=-1)
            embedded_seq = self._embedding_layer(inputs)
        return self.encode(embedded_seq, training, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embedding_layer": keras.saving.serialize_keras_object(self._embedding_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        embedding_layer = keras.layers.deserialize(config.pop("embedding_layer"))
        return cls(embedding_layer=embedding_layer, **config)

    def _check_embedding_layer_input_shape(self, input_shape):
        if isinstance(self._embedding_layer, keras.layers.Embedding):
            # Note: when using an embedding layer can't run the graph while using Tensorflow backend
            if len(input_shape) > 2 and input_shape[-1] > 1:
                raise ValueError(f"An embedding layer can be used only for sequences with one feature. "
                                 f"The input sequence has {input_shape[-1]} features. Use a Dense layer instead.")
            input_shape = input_shape[:2]
        return input_shape


@keras.saving.register_keras_serializable(package="SequenceDecoders", name="SequenceDecoder")
class SequenceDecoder(keras.Model):

    def __init__(self,
                 embedding_layer: keras.Layer = None,
                 sampling_scheduler: Scheduler = None,
                 name: str = "seq_decoder",
                 **kwargs):
        super(SequenceDecoder, self).__init__(name=name, **kwargs)
        self._embedding_layer = embedding_layer
        self._sampling_scheduler = sampling_scheduler

    def decode(self, input_sequence, aux_inputs, z, sampling_probability: float = 1.0, **kwargs):
        raise NotImplementedError("SequenceDecoder.decode must be overridden by subclasses.")

    def sample(self, z, seq_lengths, sampling_mode: str = "argmax", temperature: float = 1.0, **kwargs):
        raise NotImplementedError("SequenceDecoder.sample must be overridden by subclasses.")

    def call(self,
             inputs,
             current_step=None,
             sampling_mode: str = "argmax",
             temperature: float = 1.0,
             training: bool = False,
             evaluate: bool = False,
             **kwargs):
        if training or evaluate:
            input_sequence, aux_inputs, z = inputs
            sampling_probability = self._sampling_scheduler(step=current_step) \
                if training and self._sampling_scheduler else 1.0
            return self.decode(input_sequence, aux_inputs, z, sampling_probability, **kwargs)
        else:
            z, seq_length = inputs
            if len(seq_length.shape) > 1:
                seq_length = seq_length.shape[1]
            return self.sample(z, seq_length, sampling_mode, temperature, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embedding_layer": keras.saving.serialize_keras_object(self._embedding_layer),
            "sampling_scheduler": self._sampling_scheduler
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        embedding_layer = keras.saving.deserialize_keras_object(config.pop("embedding_layer"))
        return cls(embedding_layer=embedding_layer, **config)
