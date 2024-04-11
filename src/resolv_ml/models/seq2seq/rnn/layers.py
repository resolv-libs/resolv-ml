# TODO - DOC
from typing import List, Any

import keras
import keras.ops as k_ops
import numpy as np


def get_default_rnn_cell(cell_size: int, dropout: float) -> keras.layers.LSTMCell:
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
        dropout=dropout,
        recurrent_dropout=0.0,
        seed=None
    )


@keras.saving.register_keras_serializable(package="RNNLayers", name="InitialRNNCellStateFromEmbedding")
class InitialRNNCellStateFromEmbedding(keras.Layer):

    def __init__(self, cell_state_size, name="z_to_initial_state", **kwargs):
        super(InitialRNNCellStateFromEmbedding, self).__init__(name=name, **kwargs)
        self._cell_state_size = cell_state_size
        self._initial_cell_states = keras.layers.Dense(
            units=sum(self._get_flatten_state_sizes()),
            activation='tanh',
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name="dense"
        )

    def call(self, inputs, training: bool = False, *args, **kwargs):
        initial_cell_states = self._initial_cell_states(inputs, training=training)
        split_indexes = np.cumsum(self._get_flatten_state_sizes())[:-1]
        split_states = k_ops.split(initial_cell_states, indices_or_sections=split_indexes, axis=-1)
        packed_states = [[split_states[i], split_states[i + 1]] for i in range(0, len(split_states), 2)]
        return packed_states

    def compute_output_shape(self, input_shape):
        return self._cell_state_size

    def get_config(self):
        base_config = super().get_config()
        config = {
            "cell_state_size": self._cell_state_size,
            "initial_cell_states": keras.saving.serialize_keras_object(self._initial_cell_states),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        stacked_rnn_cells = keras.layers.deserialize(config.pop("stacked_rnn_cells"))
        return cls(stacked_rnn_cells, **config)

    def _get_flatten_state_sizes(self):
        return np.ravel(self._cell_state_size)


@keras.saving.register_keras_serializable(package="RNNLayers", name="StackedBidirectionalRNN")
class StackedBidirectionalRNN(keras.Layer):

    def __init__(self,
                 layers_sizes: List[int],
                 rnn_cell: Any = None,
                 merge_modes: List[str] = None,
                 return_sequences: bool = False,
                 return_state: bool = True,
                 dropout: float = 0.0,
                 name="stacked_bidirectional_lstm", **kwargs):
        super(StackedBidirectionalRNN, self).__init__(name=name, **kwargs)
        if not layers_sizes:
            raise ValueError("No layers provided. Please provide a list containing the layers sizes.")
        self._return_sequences = return_sequences
        self._return_state = return_state
        self._layers: List[keras.layers.Bidirectional] = []
        for i, layer_size in enumerate(layers_sizes):
            is_output_layer = (i == len(layers_sizes) - 1)
            is_return_sequences_layer = return_sequences if is_output_layer else True
            self._layers.append(
                keras.layers.Bidirectional(
                    layer=keras.layers.RNN(
                        rnn_cell if rnn_cell else get_default_rnn_cell(layer_size, dropout),
                        return_sequences=is_return_sequences_layer,
                        return_state=True,
                        go_backwards=False,
                        stateful=False,
                        unroll=False,
                        zero_output_for_mask=False
                    ),
                    merge_mode=merge_modes[i] if merge_modes else "concat",
                    weights=None,
                    backward_layer=None
                )
            )

    @property
    def state_size(self):
        return [self._layers[0].forward_layer.state_size, self._layers[0].backward_layer.state_size]

    def build(self, input_shape):
        super().build(input_shape)
        for i, layer in enumerate(self._layers):
            layer.build(input_shape)
            output_shape = layer.compute_output_shape(input_shape)
            input_shape, _ = output_shape[0], output_shape[1:]

    def call(self, inputs, training: bool = False, **kwargs):
        initial_states_fw = kwargs.get("initial_states_fw")
        initial_states_bw = kwargs.get("initial_states_bw")
        if (initial_states_fw and
                (not isinstance(initial_states_fw, list) or len(initial_states_fw) != len(self._layers))):
            raise ValueError("initial_states_fw must be a list of state tensors (one per layer).")
        if (initial_states_bw and
                (not isinstance(initial_states_bw, list) or len(initial_states_bw) != len(self._layers))):
            raise ValueError("initial_states_bw must be a list of state tensors (one per layer).")

        layers_states = []
        prev_output = inputs
        for layer_idx, layer in enumerate(self._layers):
            if initial_states_fw is None or initial_states_bw is None:
                initial_state = None
            else:
                initial_state = k_ops.concatenate([initial_states_fw[layer_idx], initial_states_bw[layer_idx]])
            prev_output, fw_h, fw_c, bw_h, bw_c = layer(prev_output, initial_state=initial_state,
                                                        mask=kwargs.get("mask"), training=training)
            state_h = k_ops.concatenate([fw_h, bw_h], axis=-1)
            state_c = k_ops.concatenate([fw_c, bw_c], axis=-1)
            layers_states += [(state_h, state_c)]
        return [prev_output, *layers_states] if self._return_state else prev_output

    def compute_output_shape(self, input_shape, initial_state_shape=None):
        output_shape = input_shape
        state_shape = None
        for i, layer in enumerate(self._layers):
            output_shape = layer.compute_output_shape(output_shape, initial_state_shape)
            output_shape, state_shape = output_shape[0], output_shape[1:]
        if self._return_state:
            state_shape = list(state_shape[0])
            state_shape[-1] *= 2
            state_shape = (tuple(state_shape),) + (tuple(state_shape),)
            return (output_shape,) + state_shape
        return output_shape

    def get_initial_state(self, batch_size):
        return [self._layers[0].forward_layer.get_initial_state(batch_size),
                self._layers[0].backward_layer.state_size.get_initial_state(batch_size)]

    def get_config(self):
        base_config = super().get_config()
        config = {
            "return_sequences": self._return_sequences,
            "return_state": self._return_state,
            "layers": [keras.saving.serialize_keras_object(layer) for layer in self._layers],
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        layers = [keras.layers.deserialize(layer) for layer in config.pop("layers")]
        return cls(layers, **config)
