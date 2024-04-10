# TODO - DOC
import random
from typing import Tuple, List, Any

import keras
import keras.ops as k_ops

from . import layers as rnn_layers
from ..base import SequenceDecoder


@keras.saving.register_keras_serializable(package="SequenceDecoders", name="RNNAutoregressiveDecoder")
class RNNAutoregressiveDecoder(SequenceDecoder):

    def __init__(self,
                 dec_rnn_sizes: List[int],
                 num_classes: int,
                 rnn_cell: Any = None,
                 output_projection_layer: keras.Layer = None,
                 embedding_layer: keras.layers.Embedding = None,
                 sampling_mode: str = "argmax",
                 name: str = "rnn_decoder",
                 **kwargs):
        super(RNNAutoregressiveDecoder, self).__init__(
            num_classes=num_classes,
            embedding_layer=embedding_layer,
            sampling_mode=sampling_mode,
            name=name,
            **kwargs
        )
        self._stacked_rnn_cells = rnn_layers.StackedRNN(
            layers_sizes=dec_rnn_sizes,
            rnn_cell=rnn_cell,
            return_sequences=False,
            return_state=True,
            name="stacked_rnn_cells"
        )
        self._initial_state_layer = rnn_layers.InitialRNNCellStateFromEmbedding(
            cell_state_size=self._stacked_rnn_cells.state_size,
            name="z_to_initial_state"
        )
        self._output_projection = output_projection_layer

    def build(self, input_shape):
        input_sequence_shape, _, z_shape = input_shape
        super().build(input_sequence_shape)
        batch_size, sequence_length, features_length = input_sequence_shape
        if self._output_projection:
            projection_layer_out_shape = self._output_projection.compute_output_shape(input_shape)
            if projection_layer_out_shape != (batch_size, 1, self._num_classes):
                raise ValueError(f"Given projection layer output shape {projection_layer_out_shape} is not compatible "
                                 f"with the decoder. Output shape must be (batch_size, 1, {self._num_classes}).")
        else:
            self._output_projection = keras.layers.Dense(
                units=self._num_classes,
                activation="linear",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="output_projection"
            )
        rnn_input_shape = batch_size, 1, (self._get_decoder_input_size(input_sequence_shape) + z_shape[1])
        self._stacked_rnn_cells.build(input_shape=rnn_input_shape)
        self._initial_state_layer.build(input_shape=z_shape)
        self._output_projection.build(input_shape=self._stacked_rnn_cells.compute_output_shape(rnn_input_shape)[0])

    def decode(self, input_sequence, aux_inputs, z, teacher_force_probability, training: bool = False, **kwargs):
        # TODO - Inference mode
        batch_size, sequence_length, features_length = input_sequence.shape if training else (1, 1, 1)
        initial_state = self._initial_state_layer(z, training=training)
        z = k_ops.expand_dims(z, axis=1)
        decoder_input = k_ops.zeros(shape=(batch_size, self._get_decoder_input_size(input_sequence.shape)))
        output_sequence_probabilities = []
        for i in range(sequence_length):
            decoder_input = k_ops.expand_dims(decoder_input, axis=1)
            decoder_input = k_ops.concatenate([decoder_input, z], axis=-1)
            dec_emb_output, *initial_state = self._stacked_rnn_cells(decoder_input,
                                                                     initial_state=initial_state,
                                                                     training=training)
            output_logits = self._output_projection(dec_emb_output, training=training)
            token_probabilities = keras.ops.softmax(output_logits)
            output_sequence_probabilities.append(token_probabilities)
            # Compute token for auto-regression
            # TODO - In order to compute the sparse categorical loss I need to return the probabilities for each token
            #  of the sequence, should I use argmax token anyway as autoregression input
            if self._sampling_mode == "argmax":
                predicted_token = keras.ops.argmax(token_probabilities, axis=-1)
            else:
                predicted_token = keras.ops.argmax(token_probabilities, axis=-1)
            ar_token = input_sequence[:, i, :] if random.random() < teacher_force_probability else predicted_token
            decoder_input = self._embedding_layer(ar_token)
        output_sequence_probabilities = k_ops.stack(output_sequence_probabilities, axis=1)
        return output_sequence_probabilities

    def compute_output_shape(self, input_shape):
        input_sequence_shape = input_shape[0]
        return input_sequence_shape[0], input_sequence_shape[1], self._num_classes

    def get_config(self):
        base_config = super().get_config()
        config = {
            "stacked_rnn_cells": keras.saving.serialize_keras_object(self._stacked_rnn_cells),
            "initial_state_layer": keras.saving.serialize_keras_object(self._initial_state_layer),
            "output_projection": keras.saving.serialize_keras_object(self._output_projection)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        stacked_rnn_cells = keras.layers.deserialize(config.pop("stacked_rnn_cells"))
        initial_state_layer = keras.layers.deserialize(config.pop("initial_state_layer"))
        output_projection = keras.layers.deserialize(config.pop("output_projection"))
        return cls(stacked_rnn_cells, initial_state_layer, output_projection, **config)

    def _get_decoder_input_size(self, input_sequence_shape):
        if self._embedding_layer:
            return self._embedding_layer.compute_output_shape(input_sequence_shape)[-1]
        else:
            return input_sequence_shape[-1]


class HierarchicalRNNDecoder(SequenceDecoder):

    def __init__(self,
                 level_lengths: List[int],
                 core_decoder: SequenceDecoder,
                 dec_rnn_sizes: List[int],
                 embedding_layer: keras.layers.Embedding = None,
                 logits_sampling_mode: str = "argmax",
                 dropout: float = 0.0,
                 name="hierarchical_decoder",
                 **kwargs):
        super(HierarchicalRNNDecoder, self).__init__(
            embedding_layer=embedding_layer,
            logits_sampling_mode=logits_sampling_mode,
            name=name,
            **kwargs
        )
        self._level_lengths = level_lengths
        self._num_levels = len(level_lengths) - 1  # subtract 1 for the core decoder level
        self._core_decoder = core_decoder
        self._hierarchical_initial_states: List[rnn_layers.InitialRNNCellStateFromEmbedding] = []
        self._stacked_hierarchical_rnn_cells: List[rnn_layers.StackedRNN] = []
        for level_idx in range(self._num_levels):
            level_rnn_cells = rnn_layers.StackedRNN(
                layers_sizes=dec_rnn_sizes,
                return_sequences=False,
                return_state=True,
                dropout=dropout,
                name=f"level_{level_idx}_rnn"
            )
            self._hierarchical_initial_states.append(
                rnn_layers.InitialRNNCellStateFromEmbedding(
                    rnn_layer=level_rnn_cells,
                    name=f"level_{level_idx}_emb_to_initial_state"
                )
            )
            self._stacked_hierarchical_rnn_cells.append(level_rnn_cells)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        super().build(input_shape)
        # Check if given hierarchical level lengths are compatible with input sequence.
        input_sequence_shape = input_shape
        input_sequence_length = input_sequence_shape[1]
        level_lengths_prod = k_ops.prod(self._level_lengths)
        if input_sequence_length != level_lengths_prod:
            raise ValueError(f"The product of the HierarchicalLSTMDecoder level lengths {level_lengths_prod} must "
                             f"equal the input sequence length {input_sequence_length}.")
        # Build core decoder
        self._core_decoder.build(input_shape)
        # Build hierarchical layers.
        for level_idx in range(self._num_levels):
            self._hierarchical_initial_states[level_idx].build(input_shape)
            self._stacked_hierarchical_rnn_cells[level_idx].build(input_shape)

    def decode(self, input_sequence, aux_inputs, z, teacher_force_probability, training: bool = False, **kwargs):

        def base_decode(embedding, path: List[int] = None):
            """Base function for hierarchical decoder."""
            base_input_sequence = hierarchy_input_sequence[path]
            decoded_sequence = self._core_decoder(base_input_sequence, embedding, teacher_force_probability)
            output_sequence.append(decoded_sequence)

        def recursive_decode(embedding, path: List[int] = None):
            """Recursive hierarchical decode function."""
            path = path or []
            level = len(path)

            if level == self._num_levels:
                return base_decode(embedding, path)

            initial_state = self._hierarchical_initial_states[level](embedding, training=training)
            level_input = k_ops.zeros(shape=(batch_size, 1))
            num_steps = self._level_lengths[level]
            for idx in range(num_steps):
                output, *initial_state = self._stacked_hierarchical_rnn_cells[level](level_input,
                                                                                     initial_states=initial_state)
                recursive_decode(output, path + [idx])

        batch_size = input_sequence.shape[0]
        hierarchy_input_sequence = self._reshape_to_hierarchy(input_sequence)
        output_sequence = []
        recursive_decode(z)
        return k_ops.stack(output_sequence, axis=1)

    def _reshape_to_hierarchy(self, tensor):
        """Reshapes `tensor` so that its initial dimensions match the hierarchy."""
        tensor_shape = tensor.shape.as_list()
        tensor_rank = len(tensor_shape)
        batch_size = tensor_shape[0]
        hierarchy_shape = [batch_size] + self._level_lengths[:-1]  # Exclude the final, core decoder length.
        if tensor_rank == 3:
            hierarchy_shape += [-1] + tensor_rank[2:]
        elif tensor_rank != 2:
            # We only expect rank-2 for lengths and rank-3 for sequences.
            raise ValueError(f"Unexpected shape for tensor: {tensor}")
        hierarchy_tensor = k_ops.reshape(tensor, hierarchy_shape)
        # Move the batch dimension to after the hierarchical dimensions.
        perm = list(range(len(hierarchy_shape)))
        perm.insert(self._num_levels, perm.pop(0))
        return k_ops.transpose(hierarchy_tensor, perm)
