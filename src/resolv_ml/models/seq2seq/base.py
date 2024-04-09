# TODO - DOC
from abc import ABC, abstractmethod

import keras

from .helpers import training as training_helper


@keras.saving.register_keras_serializable(package="Seq2SeqBase", name="SequenceEncoder")
class SequenceEncoder(ABC, keras.Model):

    def __init__(self,
                 embedding_layer: keras.layers.Embedding = None,
                 name: str = "seq_encoder",
                 **kwargs):
        super(SequenceEncoder, self).__init__(name=name, **kwargs)
        self._embedding_layer = embedding_layer

    @abstractmethod
    def encode(self, inputs, training: bool = False, **kwargs):
        pass

    def build(self, input_shape):
        super().build(input_shape)
        self._embedding_layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        # TODO - What about multi featured sequences? Not working with graph execution
        inputs = keras.ops.squeeze(inputs, axis=-1)
        embedded_seq = self._embedding_layer(inputs) if self._embedding_layer else inputs
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
        return cls(embedding_layer, **config)


@keras.saving.register_keras_serializable(package="Seq2SeqBase", name="SequenceDecoder")
class SequenceDecoder(ABC, keras.Model):

    def __init__(self,
                 embedding_layer: keras.layers.Embedding = None,
                 logits_sampling_mode: str = "argmax",
                 sampling_schedule: str = "constant",
                 sampling_rate: float = 0.0,
                 name: str = "seq_decoder",
                 **kwargs):
        super(SequenceDecoder, self).__init__(name=name, **kwargs)
        self._embedding_layer = embedding_layer
        self._logits_sampling_mode = logits_sampling_mode
        self._sampling_schedule = sampling_schedule
        self._sampling_rate = sampling_rate

    @abstractmethod
    def decode(self, input_sequence, aux_inputs, z, teacher_force_probability, training: bool = False, **kwargs):
        pass

    def build(self, input_shape):
        super().build(input_shape)
        self._embedding_layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        input_sequence, aux_inputs, z = inputs
        teacher_force_probability = training_helper.get_sampling_probability(
            sampling_schedule=self._sampling_schedule,
            sampling_rate=self._sampling_rate,
            step=kwargs.get("iterations", 0),
            training=training
        )
        return self.decode(input_sequence, aux_inputs, z, teacher_force_probability, training=training, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embedding_layer": keras.saving.serialize_keras_object(self._embedding_layer),
            "logits_sampling_mode": self._logits_sampling_mode,
            "sampling_schedule": self._sampling_schedule,
            "sampling_rate": self._sampling_rate
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        embedding_layer = keras.layers.deserialize(config.pop("embedding_layer"))
        return cls(embedding_layer, **config)
