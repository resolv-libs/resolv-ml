# TODO - DOC
from typing import List, Any, Union

import keras
import numpy as np
from keras import ops as k_ops


def get_default_rnn_cell(cell_size: int, dropout: float, name: str = "lstm") -> keras.layers.LSTMCell:
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
        seed=None,
        name=name
    )


@keras.saving.register_keras_serializable(package="RNNLayers", name="InitialRNNCellStateFromEmbedding")
class InitialRNNCellStateFromEmbedding(keras.Layer):

    def __init__(self, cell_state_sizes, name="z_to_initial_state", **kwargs):
        super(InitialRNNCellStateFromEmbedding, self).__init__(name=name, **kwargs)
        if (not cell_state_sizes or
                not isinstance(cell_state_sizes, list) or
                any(not isinstance(x, list) for x in cell_state_sizes) or
                any(any(not isinstance(x, int) for x in y) for y in cell_state_sizes)):
            raise ValueError("cell_state_sizes must be a non empty list containing the sizes for RNN cells.")

        self._cell_state_sizes = cell_state_sizes
        self._initial_cell_states = keras.layers.Dense(
            units=sum(self._get_flatten_state_sizes()),
            activation='tanh',
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name="dense"
        )

    def build(self, input_shape):
        super().build(input_shape)
        self._initial_cell_states.build(input_shape)

    def call(self, inputs, training: bool = False, *args, **kwargs):
        initial_cell_states = self._initial_cell_states(inputs, training=training)
        split_indexes = np.cumsum(self._get_flatten_state_sizes())[:-1]
        split_states = k_ops.split(initial_cell_states, indices_or_sections=split_indexes, axis=-1)
        packed_states = [[split_states[i], split_states[i + 1]] for i in range(0, len(split_states), 2)]
        return packed_states

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        out_shape = tuple((batch_size, size) for size in self._get_flatten_state_sizes())
        return out_shape

    def _get_flatten_state_sizes(self):
        return [x for y in self._cell_state_sizes for x in y]


@keras.saving.register_keras_serializable(package="RNNLayers", name="StackedBidirectionalRNN")
class StackedBidirectionalRNN(keras.Layer):

    def __init__(self,
                 layers_sizes: List[int],
                 rnn_cell: Any = None,
                 merge_modes: Union[List[str], str] = "concat",
                 return_sequences: bool = False,
                 return_state: bool = True,
                 dropout: float = 0.0,
                 name="stacked_bidirectional_lstm", **kwargs):
        super(StackedBidirectionalRNN, self).__init__(name=name, **kwargs)
        if not layers_sizes or any(not isinstance(x, int) for x in layers_sizes):
            raise ValueError("layers_sizes must be a non empty list containing the sizes for RNN cells as integers.")

        if merge_modes:
            if isinstance(merge_modes, list):
                for idx, mode in enumerate(merge_modes):
                    if idx == len(merge_modes) - 1 and not mode:
                        raise ValueError("Mode 'None' can only be specified for the last layer.")
                    if mode not in ['sum', 'mul', 'ave', 'concat', None]:
                        raise ValueError("Invalid merge mode in given merge_modes. "
                                         "Merge mode should be one of {'sum', 'mul', 'ave', 'concat', None}")
            else:
                merge_modes = [merge_modes for _ in range(len(layers_sizes))]
        else:
            merge_modes = ["concat" for _ in range(len(layers_sizes) - 1)] + [None]

        self._layers_sizes = layers_sizes
        self._rnn_cell = rnn_cell
        self._merge_modes = merge_modes
        self._return_sequences = return_sequences
        self._return_state = return_state
        self._dropout = dropout
        self._bidirectional_layers: List[keras.layers.Bidirectional] = []
        for i, layer_size in enumerate(layers_sizes):
            is_output_layer = (i == len(layers_sizes) - 1)
            is_return_sequences_layer = return_sequences if is_output_layer else True
            self._bidirectional_layers.append(
                keras.layers.Bidirectional(
                    layer=keras.layers.RNN(
                        rnn_cell if rnn_cell else get_default_rnn_cell(layer_size, dropout, name=f"lstm_{i}"),
                        return_sequences=is_return_sequences_layer,
                        return_state=True,
                        go_backwards=False,
                        stateful=False,
                        unroll=False,
                        zero_output_for_mask=False,
                        name=f"rnn_{i}"
                    ),
                    merge_mode=merge_modes[i],
                    weights=None,
                    backward_layer=None,
                    name=f"bidirectional_{i}"
                )
            )

    @property
    def state_size(self):
        return [self._bidirectional_layers[0].forward_layer.state_size,
                self._bidirectional_layers[0].backward_layer.state_size]

    def build(self, input_shape):
        super().build(input_shape)
        for i, layer in enumerate(self._bidirectional_layers):
            layer.build(input_shape)
            output_shape = layer.compute_output_shape(input_shape)
            input_shape, _ = output_shape[0], output_shape[1:]

    def call(self, inputs, initial_state=None, training: bool = False, **kwargs):
        if (initial_state and
                (not isinstance(initial_state, list) or len(initial_state) != len(self._bidirectional_layers))):
            raise ValueError("initial_states_fw must be a list of state tensors (one per layer).")

        layers_states = []
        prev_output = inputs
        last_merge_mode = None
        for idx, layer in enumerate(self._bidirectional_layers):
            layer_outputs = layer(prev_output, initial_state=initial_state, mask=kwargs.get("mask"), training=training)
            if layer.merge_mode:
                prev_output, fw_h, fw_c, bw_h, bw_c = layer_outputs
            else:
                *prev_output, fw_h, fw_c, bw_h, bw_c = layer_outputs
            layers_states += [[fw_h, fw_c, bw_h, bw_c]]
            last_merge_mode = layer.merge_mode
        if last_merge_mode:
            prev_output = [prev_output]
        # TODO - add support for returning the states of all the layers when return_state=True.
        #  Remember to update also compute_output_shape and all the classes that uses this layer
        #  (in particular check methods build, compute_output_shape and call)
        return tuple(prev_output) + tuple(layers_states[-1]) if self._return_state else prev_output

    def compute_output_shape(self, inputs_shape, initial_state_shape=None):
        output_shape = inputs_shape
        state_shape = None
        merge_mode = None

        for layer in self._bidirectional_layers:
            merge_mode = layer.merge_mode
            output_shape = layer.compute_output_shape(output_shape, initial_state_shape)
            if not merge_mode:
                output_shape, state_shape = output_shape[:2], output_shape[2:]
            else:
                output_shape, state_shape = output_shape[0], output_shape[1:]

        if self._return_state:
            if not merge_mode:
                return tuple(output_shape) + state_shape
            return tuple([output_shape]) + state_shape

        return tuple(output_shape)

    def get_initial_state(self, batch_size):
        return [self._bidirectional_layers[0].forward_layer.get_initial_state(batch_size),
                self._bidirectional_layers[0].backward_layer.state_size.get_initial_state(batch_size)]

    def get_config(self):
        base_config = super().get_config()
        config = {
            "layers_sizes": self._layers_sizes,
            "rnn_cell": keras.saving.serialize_keras_object(self._rnn_cell),
            "merge_modes": self._merge_modes,
            "return_sequences": self._return_sequences,
            "return_state": self._return_state,
            "dropout": self._dropout
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        rnn_cell = keras.saving.deserialize_keras_object(config.pop("rnn_cell"))
        return cls(rnn_cell=rnn_cell, **config)
