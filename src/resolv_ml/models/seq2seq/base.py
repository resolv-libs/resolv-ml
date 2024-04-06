# TODO - DOC
from abc import ABC, abstractmethod

import keras
from .helpers import training as training_helper


class SequenceEncoder(ABC, keras.Model):

    def __init__(self, name: str = "seq_encoder", **kwargs):
        super(SequenceEncoder, self).__init__(name=name, **kwargs)

    @abstractmethod
    def encode(self, inputs, training: bool = False, **kwargs):
        pass

    def call(self, inputs, training: bool = False, **kwargs):
        return self.encode(inputs, training, **kwargs)


class SequenceDecoder(ABC, keras.Model):

    def __init__(self,
                 sampling_schedule: str = "constant",
                 sampling_rate: float = 0.0,
                 name: str = "seq_decoder",
                 **kwargs):
        super(SequenceDecoder, self).__init__(name=name, **kwargs)
        self._sampling_schedule = sampling_schedule
        self._sampling_rate = sampling_rate

    @abstractmethod
    def decode(self, input_sequence, embedding, teacher_force_probability, training: bool = False, **kwargs):
        pass

    def call(self, input_sequence, embedding, training: bool = False, **kwargs):
        teacher_force_probability = self._get_teacher_force_probability(training)
        return self.decode(input_sequence, embedding, teacher_force_probability, training=training, **kwargs)

    def _get_teacher_force_probability(self, training: bool = False):
        return training_helper.get_sampling_probability(
            sampling_schedule=self._sampling_schedule,
            sampling_rate=self._sampling_rate,
            step=self.optimizer.iterations,
            training=training
        )