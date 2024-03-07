# TODO - DOC
from typing import List, Tuple, Any

import keras


class InitialRNNCellStateFromEmbedding(keras.Layer):

    def __init__(self, layers_sizes: List[int], name="z_to_initial_state", **kwargs):
        super(InitialRNNCellStateFromEmbedding, self).__init__(name=name, **kwargs)
        self._layers_sizes = layers_sizes

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):

        def split(cell_states):
            split_indexes = [sum(flatten_state_sizes[:i+1]) - 1 for i in range(len(flatten_state_sizes))]
            return keras.ops.split(cell_states, indices_or_sections=split_indexes, axis=-1)

        def pack(cell_states):
            return list(zip(cell_states[::2], cell_states[1::2]))

        cell_state_sizes = [(layer_size, layer_size) for layer_size in self._layers_sizes]
        flatten_state_sizes = [x for cell_state_size in cell_state_sizes for x in cell_state_size]
        self._initial_cell_states = keras.layers.Dense(
            units=sum(flatten_state_sizes),
            activation='tanh',
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name=f"{self.name}/dense"
        )
        self._split_initial_cell_states = keras.layers.Lambda(split, name=f"{self.name}/split")
        self._pack_initial_cell_states = keras.layers.Lambda(pack, name=f"{self.name}/pack")

    def call(self, embedding, training=False, *args, **kwargs):
        initial_cell_states = self._initial_cell_states(embedding, training=training)
        split_initial_cell_states = self._split_initial_cell_states(initial_cell_states, training=training)
        packed_initial_cell_states = self._pack_initial_cell_states(split_initial_cell_states, training=training)
        return packed_initial_cell_states


class StackedLSTM(keras.Layer):

    def __init__(self,
                 layers_sizes: List[int],
                 return_sequences: bool = False,
                 return_state: bool = True,
                 lstm_dropout: float = 0.0,
                 name="stacked_lstm", **kwargs):
        super(StackedLSTM, self).__init__(name=name, **kwargs)
        self._stacked_rnn_sizes = layers_sizes
        self._return_sequences = return_sequences
        self._return_state = return_state
        self._dropout = lstm_dropout

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...], **kwargs):

        def build_lstm_cells(units: int):
            return keras.layers.LSTMCell(
                units=units,
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
                dropout=self._dropout,
                recurrent_dropout=0.0,
                seed=None
            )

        self._stacked_lstm_cells = keras.layers.RNN(
            [build_lstm_cells(rnn_size) for rnn_size in self._stacked_rnn_sizes],
            return_sequences=self._return_sequences,
            return_state=self._return_state,
            go_backwards=False,
            stateful=False,
            time_major=False,
            unroll=False,
            zero_output_for_mask=False,
            name="stacked_lstm_cells"
        )

    def call(self, inputs, initial_states, training: bool = False, mask=None, **kwargs) -> Any:
        return self._stacked_lstm_cells(inputs, initial_states=initial_states, training=training, mask=mask)


class StackedBidirectionalLSTM(keras.Layer):

    def __init__(self,
                 layers_sizes: List[int],
                 return_sequences: bool = False,
                 return_state: bool = True,
                 lstm_dropout: float = 0.0,
                 name="stacked_bidirectional_lstm", **kwargs):
        super(StackedBidirectionalLSTM, self).__init__(name=name, **kwargs)
        if not layers_sizes:
            raise ValueError("No layers provided. Please provide a list containing the layers sizes.")
        self._layers_sizes = layers_sizes
        self._return_sequences = return_sequences
        self._return_state = return_state
        self._dropout = lstm_dropout
        self._layers: List[keras.layers.Bidirectional] = []

    def build(self, input_shape: Tuple[int, ...]):
        for i, size in enumerate(self._layers_sizes):
            is_output_layer = i == len(self._layers_sizes) - 1
            self._layers.append(
                keras.layers.Bidirectional(
                    units=keras.layers.LSTM(
                        units=size,
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
                        activity_regularizer=None,
                        kernel_constraint=None,
                        recurrent_constraint=None,
                        bias_constraint=None,
                        dropout=self._dropout,
                        recurrent_dropout=0.0,
                        seed=None,
                        return_sequences=self._return_sequences if is_output_layer else True,
                        return_state=True,
                        go_backwards=False,
                        stateful=False,
                        unroll=False
                    ),
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
            # Beware that when passing the `initial_state` argument during the call of this layer, the first half in the
            # list of elements in the `initial_state` list will be passed to the forward RNN call and the last half in
            # the list of elements will be passed to the backward RNN call.
            initial_states = keras.layers.Concatenate(initial_states_fw[layer_idx], initial_states_bw[layer_idx])
            output, fw_h, fw_c, bw_h, bw_c = self._layers[layer_idx](prev_output, initial_states=initial_states,
                                                                     training=training, mask=mask)
            state_h = keras.layers.Concatenate([fw_h, bw_h])
            state_c = keras.layers.Concatenate([fw_c, bw_c])
            prev_output = output
            hidden_states.append(state_h)
            cell_states.append(state_c)

        return prev_output, hidden_states[-1], cell_states[-1] if self._return_state else prev_output
