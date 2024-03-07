# TODO - DOC
from typing import List, Any, Tuple

from ..base import VAEEncoder
from resolv_ml.utilities.nn import rnn as rnn_utils


class LSTMEncoder(VAEEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 lstm_dropout: float = 0.0,
                 name: str = "vae/lstm_encoder",
                 **kwargs):
        super(LSTMEncoder, self).__init__(name=name, **kwargs)
        self._stacked_rnn_sizes = enc_rnn_sizes
        self._dropout = lstm_dropout

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...], **kwargs):
        self._stacked_lstm_cells = rnn_utils.StackedLSTM(
            layers_sizes=self._stacked_rnn_sizes,
            return_sequences=False,
            return_state=True,
            lstm_dropout=self._dropout,
            name=f'{self.name}/stacked_lstm_cells'
        )

    def _encode(self, inputs, training: bool = False, **kwargs) -> Any:
        _, hidden_state, cell_state = self._stacked_lstm_cells(inputs, training=training)
        return hidden_state


class BidirectionalLstmEncoder(VAEEncoder):

    def __init__(self,
                 enc_rnn_sizes: List[int],
                 lstm_dropout: float = 0.0,
                 name="vae/bidirectional_lstm_encoder",
                 **kwargs):
        super(BidirectionalLstmEncoder, self).__init__(name=name, **kwargs)
        self._rnn_sizes = enc_rnn_sizes
        self._dropout = lstm_dropout

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...], **kwargs):
        self.stacked_bidirectional_lstm_layers = rnn_utils.StackedBidirectionalLSTM(
            layers_sizes=self._rnn_sizes,
            return_sequences=False,
            return_state=True,
            lstm_dropout=self._dropout,
            name=f'{self.name}/stacked_bidirectional_lstm_layers'
        )

    def _encode(self, inputs, training: bool = False, **kwargs) -> Any:
        _, hidden_state, cell_state = self.stacked_bidirectional_lstm_layers(inputs, training=training)
        return hidden_state
