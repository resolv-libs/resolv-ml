# TODO - DOC
from typing import List, Tuple

from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell

from . import layers as rnn_layers
from ..base import SequenceEncoder


class RNNEncoder(SequenceEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 rnn_cell: DropoutRNNCell = None,
                 dropout: float = 0.0,
                 name: str = "rnn_encoder",
                 **kwargs):
        super(RNNEncoder, self).__init__(name=name, **kwargs)
        self._stacked_rnn_sizes = enc_rnn_sizes
        self._rnn_cell = rnn_cell
        self._dropout = dropout

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...], **kwargs):
        self._stacked_rnn_cells = rnn_layers.StackedRNN(
            layers_sizes=self._stacked_rnn_sizes,
            rnn_cell=self._rnn_cell,
            return_sequences=False,
            return_state=True,
            lstm_dropout=self._dropout,
            name=f'{self.name}/stacked_rnn_cells'
        )

    def encode(self, inputs, training: bool = False, **kwargs):
        _, hidden_state, cell_state = self._stacked_rnn_cells(inputs, training=training)
        return hidden_state


class BidirectionalRNNEncoder(SequenceEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 rnn_cell: DropoutRNNCell = None,
                 dropout: float = 0.0,
                 name="bidirectional_rnn_encoder",
                 **kwargs):
        super(BidirectionalRNNEncoder, self).__init__(name=name, **kwargs)
        self._rnn_sizes = enc_rnn_sizes
        self._rnn_cell = rnn_cell
        self._dropout = dropout

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...], **kwargs):
        self.stacked_bidirectional_lstm_layers = rnn_layers.StackedBidirectionalRNN(
            layers_sizes=self._rnn_sizes,
            rnn_cell=self._rnn_cell,
            return_sequences=False,
            return_state=True,
            lstm_dropout=self._dropout,
            name=f'{self.name}/stacked_bidirectional_rnn'
        )

    def encode(self, inputs, training: bool = False, **kwargs):
        _, hidden_state, cell_state = self.stacked_bidirectional_lstm_layers(inputs, training=training)
        return hidden_state
