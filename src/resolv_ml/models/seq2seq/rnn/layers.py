# TODO - DOC
from typing import List, Tuple, Any

import keras
import keras.ops as k_ops
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell


def get_default_rnn_cell(cell_size: int):
    return keras.layers.LSTMCell(
                units=cell_size,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.0,
                recurrent_dropout=0.0,
                seed=None
            )


class InitialRNNCellStateFromEmbedding(keras.Layer):

    class SplitAndPackCellStates(keras.Layer):

        def __init__(self, split_indexes: List[int], name="split_and_pack"):
            super(InitialRNNCellStateFromEmbedding.SplitAndPackCellStates, self).__init__(name=name)
            self._split_indexes = split_indexes

        def call(self, cell_states, **kwargs):
            split_indexes = [sum(flatten_state_sizes[:i + 1]) - 1 for i in range(len(flatten_state_sizes))]
            split_cell_states = k_ops.split(cell_states, indices_or_sections=self._split_indexes, axis=-1)
            even_indices = k_ops.arange(start=0, stop=cell_states_size, step=2)
            odd_indices = k_ops.arange(start=1, stop=cell_states_size, step=2)
            even_cell_states = k_ops.take(cell_states, indices=even_indices, axis=-1)
            odd_cell_states = k_ops.take(cell_states, indices=odd_indices, axis=-1)
            packed_states = k_ops.stack([even_cell_states, odd_cell_states], axis=-1)
            return packed_states

    def __init__(self, layers_sizes: List[int], name="z_to_initial_state", **kwargs):
        super(InitialRNNCellStateFromEmbedding, self).__init__(name=name, **kwargs)
        self._layers_sizes = layers_sizes

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        self._initial_cell_states = keras.layers.Dense(
            units=sum(self._get_flatten_states_sizes()),
            activation='tanh',
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name=f"{self.name}/dense"
        )
        self._split_and_pack_cell_states = self.SplitAndPackCellStates(name=f"{self.name}/split_and_pack")

    def call(self, embedding, training=False, *args, **kwargs):
        initial_cell_states = self._initial_cell_states(embedding, training=training)
        return self._split_and_pack_cell_states(initial_cell_states, training=training)

    def _get_flatten_states_sizes(self):
        return [item for item in self._layers_sizes for _ in range(2)]


class StackedRNN(keras.Layer):

    def __init__(self,
                 layers_sizes: List[int],
                 rnn_cell: DropoutRNNCell = None,
                 return_sequences: bool = False,
                 return_state: bool = True,
                 dropout: float = 0.0,
                 go_backwards: bool = False,
                 stateful: bool = False,
                 time_major: bool = False,
                 unroll: bool = False,
                 zero_output_for_mask: bool = False,
                 name="stacked_rnn", **kwargs):
        super(StackedRNN, self).__init__(name=name, **kwargs)
        self._stacked_rnn_sizes = layers_sizes
        self._rnn_cell = rnn_cell
        self._return_sequences = return_sequences
        self._return_state = return_state
        self._dropout = dropout
        self._go_backwards = go_backwards
        self._stateful = stateful
        self._time_major = time_major
        self._unroll = unroll
        self._zero_output_for_mask = zero_output_for_mask

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...], **kwargs):

        def build_rnn_cells(units: int):
            rnn_cell = self._rnn_cell if self._rnn_cell else get_default_rnn_cell(units)
            rnn_cell.dropout = self._dropout
            return rnn_cell

        self._stacked_rnn_cells = keras.layers.RNN(
            [build_rnn_cells(rnn_size) for rnn_size in self._stacked_rnn_sizes],
            return_sequences=self._return_sequences,
            return_state=self._return_state,
            go_backwards=self._go_backwards,
            stateful=self._stateful,
            time_major=self._time_major,
            unroll=self._unroll,
            zero_output_for_mask=self._zero_output_for_mask,
            name="stacked_rnn_cells"
        )

    def call(self, inputs, initial_states, training: bool = False, mask=None, **kwargs) -> Any:
        return self._stacked_rnn_cells(inputs, initial_states=initial_states, training=training, mask=mask)


class StackedBidirectionalRNN(keras.Layer):

    def __init__(self,
                 layers_sizes: List[int],
                 rnn_cell: DropoutRNNCell = None,
                 return_sequences: bool = False,
                 return_state: bool = True,
                 dropout: float = 0.0,
                 name="stacked_bidirectional_lstm", **kwargs):
        super(StackedBidirectionalRNN, self).__init__(name=name, **kwargs)
        if not layers_sizes:
            raise ValueError("No layers provided. Please provide a list containing the layers sizes.")
        self._layers_sizes = layers_sizes
        self._rnn_cell = rnn_cell
        self._return_sequences = return_sequences
        self._return_state = return_state
        self._dropout = dropout
        self._layers: List[keras.layers.Bidirectional] = []

    def _get_base_rnn_layer(self, units: int, return_sequences: bool = True):
        rnn_cell = self._rnn_cell if self._rnn_cell else get_default_rnn_cell(units)
        rnn_cell.dropout = self._dropout
        return keras.layers.RNN(
            rnn_cell,
            return_sequences=return_sequences,
            return_state=True,
            go_backwards=False,
            stateful=False,
            time_major=False,
            unroll=False,
            zero_output_for_mask=False
        )

    def build(self, input_shape: Tuple[int, ...]):
        for i, layer_size in enumerate(self._layers_sizes):
            is_output_layer = i == len(self._layers_sizes) - 1
            is_return_sequences_layer = self._return_sequences if is_output_layer else True
            base_rnn_units = self._get_base_rnn_layer(layer_size, return_sequences=is_return_sequences_layer)
            self._layers.append(
                keras.layers.Bidirectional(
                    units=base_rnn_units,
                    merge_mode="concat",
                    weights=None,
                    backward_layer=None
                )
            )

    def call(self, inputs, initial_states_fw, initial_states_bw, training: bool = False, mask=None, **kwargs) -> Any:
        if (initial_states_fw and
                (not isinstance(initial_states_fw, list) or len(initial_states_fw) != len(self._layers))):
            raise ValueError("initial_states_fw must be a list of state tensors (one per layer).")

        if (initial_states_bw and
                (not isinstance(initial_states_bw, list) or len(initial_states_bw) != len(self._layers))):
            raise ValueError("initial_states_bw must be a list of state tensors (one per layer).")
        hidden_states = []
        cell_states = []
        prev_output = inputs
        for layer_idx, layer in enumerate(self._layers):
            initial_states = keras.layers.Concatenate(initial_states_fw[layer_idx], initial_states_bw[layer_idx])
            output, fw_h, fw_c, bw_h, bw_c = self._layers[layer_idx](prev_output, initial_states=initial_states,
                                                                     training=training, mask=mask)
            state_h = keras.layers.Concatenate([fw_h, bw_h])
            state_c = keras.layers.Concatenate([fw_c, bw_c])
            prev_output = output
            hidden_states.append(state_h)
            cell_states.append(state_c)

        return (prev_output, hidden_states[-1], cell_states[-1]) if self._return_state else prev_output
