# TODO - DOC
from typing import List, Tuple, Any

import keras

from . import layers as rnn_layers
from ..base import SequenceEncoder


@keras.saving.register_keras_serializable(package="SequenceEncoders", name="RNNEncoder")
class RNNEncoder(SequenceEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 rnn_cell: Any = None,
                 embedding_layer: keras.layers.Embedding = None,
                 dropout: float = 0.0,
                 name: str = "rnn_encoder",
                 **kwargs):
        super(RNNEncoder, self).__init__(embedding_layer=embedding_layer, name=name, **kwargs)
        self._stacked_rnn_cells = rnn_layers.StackedRNN(
            layers_sizes=enc_rnn_sizes,
            rnn_cell=rnn_cell,
            return_sequences=False,
            return_state=True,
            dropout=dropout,
            name='stacked_rnn_cells'
        )

    @property
    def state_size(self):
        return self._stacked_rnn_cells.state_size

    def build(self, input_shape):
        super().build(input_shape)
        self._stacked_rnn_cells.build(input_shape)

    def encode(self, inputs, training: bool = False, **kwargs):
        _, hidden_state, _ = self._stacked_rnn_cells(inputs, training=training, **kwargs)
        return hidden_state

    def compute_output_shape(self, input_shape):
        return self._stacked_rnn_cells.compute_output_shape(input_shape)

    def get_initial_state(self, batch_size):
        return self._stacked_rnn_cells.get_initial_state(batch_size)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "stacked_rnn_cells": keras.saving.serialize_keras_object(self._stacked_rnn_cells),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        stacked_rnn_cells = keras.saving.deserialize_keras_object(config.pop("stacked_rnn_cells"))
        return cls(stacked_rnn_cells, **config)


@keras.saving.register_keras_serializable(package="SequenceEncoders", name="BidirectionalRNNEncoder")
class BidirectionalRNNEncoder(SequenceEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 rnn_cell: Any = None,
                 embedding_layer: keras.layers.Embedding = None,
                 dropout: float = 0.0,
                 name="bidirectional_rnn_encoder",
                 **kwargs):
        super(BidirectionalRNNEncoder, self).__init__(embedding_layer=embedding_layer, name=name, **kwargs)
        self._stacked_bidirectional_rnn_layers = rnn_layers.StackedBidirectionalRNN(
            layers_sizes=enc_rnn_sizes,
            rnn_cell=rnn_cell,
            return_sequences=False,
            return_state=True,
            dropout=dropout,
            name='stacked_bidirectional_rnn'
        )

    def build(self, input_shape):
        input_shape = (input_shape[0], input_shape[1])  # TODO - What about multi featured sequences
        super().build(input_shape)
        embedding_output_shape = self._embedding_layer.compute_output_shape(input_shape)
        self._stacked_bidirectional_rnn_layers.build(embedding_output_shape)

    def encode(self, inputs, training: bool = False, **kwargs):
        _, hidden_state, _ = self._stacked_bidirectional_rnn_layers(inputs, training=training, **kwargs)
        return hidden_state

    def compute_output_shape(self, input_shape):
        rnn_output_shape = self._stacked_bidirectional_rnn_layers.compute_output_shape(input_shape)
        return rnn_output_shape[1]

    def get_initial_state(self, batch_size):
        return self._stacked_bidirectional_rnn_layers.get_initial_state(batch_size)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "stacked_bidirectional_rnn_layers": keras.saving.serialize_keras_object(
                self._stacked_bidirectional_rnn_layers
            )
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        stacked_bidirectional_rnn_layers = keras.saving.deserialize_keras_object(
            config.pop("stacked_bidirectional_rnn_layers")
        )
        return cls(stacked_bidirectional_rnn_layers, **config)
