# TODO - DOC
from typing import List, Any

import keras

from . import layers as rnn_layers
from ..base import SequenceEncoder


@keras.saving.register_keras_serializable(package="SequenceEncoders", name="RNNEncoder")
class RNNEncoder(SequenceEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 rnn_cell: Any = None,
                 embedding_layer: keras.Layer = None,
                 dropout: float = 0.0,
                 name: str = "rnn_encoder",
                 **kwargs):
        super(RNNEncoder, self).__init__(embedding_layer=embedding_layer, name=name, **kwargs)
        self._enc_rnn_sizes = enc_rnn_sizes
        self._rnn_cell = rnn_cell
        self._dropout = dropout
        self._stacked_rnn_cells = keras.layers.RNN(
            cell=rnn_cell if rnn_cell else [rnn_layers.get_default_rnn_cell(size, dropout, name=f"lstm_cell_{i}")
                                            for i, size in enumerate(enc_rnn_sizes)],
            return_sequences=False,
            return_state=True,
            name="stacked_rnn_cells"
        )

    @property
    def state_size(self):
        return self._stacked_rnn_cells.state_size

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = self._check_embedding_layer_input_shape(input_shape)
        embedding_output_shape = self._embedding_layer.compute_output_shape(tuple(input_shape))
        self._stacked_rnn_cells.build(embedding_output_shape)

    def encode(self, inputs, training: bool = False, **kwargs):
        _, *output_states = self._stacked_rnn_cells(inputs, initial_state=None, training=training, **kwargs)
        last_hidden_state = output_states[-1][0]
        return last_hidden_state

    def compute_output_shape(self, input_shape):
        rnn_output_shape = self._stacked_rnn_cells.compute_output_shape(input_shape)
        last_layer_states_shape = rnn_output_shape[-1]
        last_layer_hidden_states_shape = last_layer_states_shape[0]
        return last_layer_hidden_states_shape

    def get_initial_state(self, batch_size):
        return self._stacked_rnn_cells.get_initial_state(batch_size)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "enc_rnn_sizes": self._enc_rnn_sizes,
            "rnn_cell": keras.saving.serialize_keras_object(self._rnn_cell),
            "dropout": self._dropout,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        rnn_cell = keras.saving.deserialize_keras_object(config.pop("rnn_cell"))
        embedding_layer = keras.layers.deserialize(config.pop("embedding_layer"))
        return cls(rnn_cell=rnn_cell, embedding_layer=embedding_layer, **config)


@keras.saving.register_keras_serializable(package="SequenceEncoders", name="BidirectionalRNNEncoder")
class BidirectionalRNNEncoder(SequenceEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 rnn_cell: Any = None,
                 embedding_layer: keras.Layer = None,
                 dropout: float = 0.0,
                 name="bidirectional_rnn_encoder",
                 **kwargs):
        super(BidirectionalRNNEncoder, self).__init__(embedding_layer=embedding_layer, name=name, **kwargs)
        self._enc_rnn_sizes = enc_rnn_sizes
        self._rnn_cell = rnn_cell
        self._dropout = dropout
        self._stacked_bidirectional_rnn_layers = rnn_layers.StackedBidirectionalRNN(
            layers_sizes=enc_rnn_sizes,
            rnn_cell=rnn_cell,
            return_sequences=False,
            return_state=True,
            dropout=dropout,
            name='stacked_bidirectional_rnn'
        )

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = self._check_embedding_layer_input_shape(input_shape)
        embedding_output_shape = self._embedding_layer.compute_output_shape(tuple(input_shape))
        self._stacked_bidirectional_rnn_layers.build(embedding_output_shape)

    def encode(self, inputs, training: bool = False, **kwargs):
        _, fw_h, _, bw_h, _ = self._stacked_bidirectional_rnn_layers(inputs, training=training, **kwargs)
        return keras.ops.concatenate([fw_h, bw_h], axis=-1)

    def compute_output_shape(self, input_shape):
        _, fw_h_shape, _, bw_h_shape, _ = self._stacked_bidirectional_rnn_layers.compute_output_shape(input_shape)
        batch_size = fw_h_shape[0]
        return batch_size, fw_h_shape[-1] + bw_h_shape[-1]

    def get_initial_state(self, batch_size):
        return self._stacked_bidirectional_rnn_layers.get_initial_state(batch_size)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "enc_rnn_sizes": self._enc_rnn_sizes,
            "rnn_cell": keras.saving.serialize_keras_object(self._rnn_cell),
            "dropout": self._dropout,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        rnn_cell = keras.saving.deserialize_keras_object(config.pop("rnn_cell"))
        embedding_layer = keras.layers.deserialize(config.pop("embedding_layer"))
        return cls(rnn_cell=rnn_cell, embedding_layer=embedding_layer, **config)
