# TODO - DOC
import random
from typing import List, Any

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
                 dropout: float = 0.0,
                 sampling_schedule: str = "constant",
                 sampling_rate: float = 0.0,
                 name: str = "rnn_decoder",
                 **kwargs):
        super(RNNAutoregressiveDecoder, self).__init__(
            num_classes=num_classes,
            embedding_layer=embedding_layer,
            sampling_schedule=sampling_schedule,
            sampling_rate=sampling_rate,
            name=name,
            **kwargs
        )
        self._stacked_rnn_cells = rnn_cell if rnn_cell else \
            [rnn_layers.get_default_rnn_cell(size, dropout) for size in dec_rnn_sizes]
        if len(dec_rnn_sizes) > 1 or (isinstance(rnn_cell, list) and len(rnn_cell) > 1):
            self._stacked_rnn_cells = keras.layers.StackedRNNCells(self._stacked_rnn_cells)
        self._initial_state_layer = rnn_layers.InitialRNNCellStateFromEmbedding(
            cell_state_sizes=self._stacked_rnn_cells.state_size,
            name="z_to_initial_state"
        )
        self._output_projection = output_projection_layer

    def build(self, input_shape):
        input_sequence_shape, _, z_shape = input_shape
        super().build(input_sequence_shape)
        if self._output_projection:
            projection_layer_out_shape = self._output_projection.compute_output_shape(input_shape)
            if projection_layer_out_shape != (input_sequence_shape[0], 1, self._num_classes):
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

    def decode(self, input_sequence, aux_inputs, z, teacher_force_probability, **kwargs):
        batch_size, sequence_length, features_length = input_sequence.shape
        initial_state = self._initial_state_layer(z, training=True)
        decoder_input = k_ops.zeros(shape=(batch_size, self._get_decoder_input_size(input_sequence.shape)))
        output_sequence_logits = []
        for i in range(sequence_length):
            decoder_input = k_ops.concatenate([decoder_input, z], axis=-1)
            predicted_token, output_logits, initial_state = self._predict_sequence_token(
                rnn_input=decoder_input,
                initial_state=initial_state,
                sampling_mode="argmax",
                training=True,
                **kwargs
            )
            output_sequence_logits.append(output_logits)
            ar_token = input_sequence[:, i, :] if random.random() < teacher_force_probability else predicted_token
            decoder_input = self._embedding_layer(ar_token)
        return k_ops.stack(output_sequence_logits, axis=1)

    def sample(self, z, sampling_mode, **kwargs):
        predicted_token, _, _ = self._predict_sequence_token(
            rnn_input=z,
            initial_state=kwargs.get("initial_state", None),
            sampling_mode=sampling_mode,
            training=False,
            **kwargs
        )
        return predicted_token

    def _predict_sequence_token(self, rnn_input, initial_state=None, temperature: float = 1.0,
                                sampling_mode: str = "argmax", training: bool = False, **kwargs):
        dec_emb_output, output_state = self._stacked_rnn_cells(rnn_input, states=initial_state, training=training)
        output_logits = self._output_projection(dec_emb_output, training=training)
        output_logits = output_logits / temperature
        token_probabilities = keras.ops.softmax(output_logits)
        if sampling_mode == "argmax":
            predicted_token = keras.ops.argmax(token_probabilities, axis=-1)
        elif sampling_mode == "categorical":
            predicted_token = keras.random.categorical(output_logits, 1, seed=kwargs.get("seed", None))
        else:
            raise NotImplementedError(f"Sampling mode {sampling_mode} is not implemented.")
        return predicted_token, output_logits, output_state

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return batch_size, None, None

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


@keras.saving.register_keras_serializable(package="SequenceDecoders", name="HierarchicalRNNDecoder")
class HierarchicalRNNDecoder(SequenceDecoder):

    def __init__(self,
                 level_lengths: List[int],
                 core_decoder: SequenceDecoder,
                 dec_rnn_sizes: List[int],
                 num_classes: int,
                 rnn_cell: Any = None,
                 dropout: float = 0.0,
                 sampling_schedule: str = "constant",
                 sampling_rate: float = 0.0,
                 name="hierarchical_decoder",
                 **kwargs):
        super(HierarchicalRNNDecoder, self).__init__(
            num_classes=num_classes,
            sampling_schedule=sampling_schedule,
            sampling_rate=sampling_rate,
            name=name,
            **kwargs
        )
        self._level_lengths = level_lengths
        self._num_levels = len(level_lengths) - 1  # subtract 1 for the core decoder level
        self._core_decoder = core_decoder
        self._hierarchical_initial_states: List[rnn_layers.InitialRNNCellStateFromEmbedding] = []
        self._stacked_hierarchical_rnn_cells: List[keras.layers.RNN] = []
        for level_idx in range(self._num_levels):
            level_rnn_cells = rnn_cell if rnn_cell else \
                [rnn_layers.get_default_rnn_cell(size, dropout) for size in dec_rnn_sizes]
            if len(dec_rnn_sizes) > 1 or (isinstance(rnn_cell, list) and len(rnn_cell) > 1):
                level_rnn_cells = keras.layers.StackedRNNCells(level_rnn_cells)
            self._hierarchical_initial_states.append(
                rnn_layers.InitialRNNCellStateFromEmbedding(
                    cell_state_sizes=level_rnn_cells.state_size,
                    name=f"level_{level_idx}_emb_to_initial_state"
                )
            )
            self._stacked_hierarchical_rnn_cells.append(level_rnn_cells)

    def build(self, input_shape):
        input_sequence_shape, _, z_shape = input_shape
        super().build(input_sequence_shape)
        batch_size, sequence_length, features_length = input_sequence_shape
        # Check if given hierarchical level lengths are compatible with input sequence.
        level_lengths_prod = k_ops.prod(self._level_lengths)
        if sequence_length != level_lengths_prod:
            raise ValueError(f"The product of the HierarchicalLSTMDecoder level lengths {level_lengths_prod} must "
                             f"equal the input sequence length {sequence_length}.")
        self._core_decoder.build(input_shape)

    def decode(self, input_sequence, aux_inputs, z, teacher_force_probability, **kwargs):

        def base_decode(embedding, path: List[int] = None):
            """Base function for hierarchical decoder."""
            base_input_sequence = hierarchy_input_sequence[path]
            decoded_sequence = self._core_decoder.decode(
                input_sequence=base_input_sequence,
                aux_inputs=aux_inputs,
                z=embedding,
                teacher_force_probability=teacher_force_probability
            )
            output_sequence.append(decoded_sequence)

        def recursive_decode(embedding, path: List[int] = None):
            """Recursive hierarchical decode function."""
            path = path or []
            level = len(path)
            if level == self._num_levels:
                return base_decode(embedding, path)
            initial_state = self._hierarchical_initial_states[level](embedding, training=True)
            level_input = k_ops.zeros(shape=(batch_size, self._get_decoder_input_size(input_sequence.shape)))
            for idx in range(self._level_lengths[level]):
                output, initial_state = self._stacked_hierarchical_rnn_cells[level](level_input,
                                                                                    states=initial_state,
                                                                                    training=True)
                recursive_decode(output, path + [idx])

        batch_size = input_sequence.shape[0]
        hierarchy_input_sequence = self._reshape_to_hierarchy(input_sequence)
        output_sequence = []
        recursive_decode(z)
        return k_ops.concatenate(output_sequence, axis=1)

    def sample(self, z, sampling_mode, **kwargs):
        pass

    def _reshape_to_hierarchy(self, tensor):
        """Reshapes `tensor` so that its initial dimensions match the hierarchy."""
        tensor_shape = tensor.shape.as_list()
        tensor_rank = len(tensor_shape)
        batch_size = tensor_shape[0]
        hierarchy_shape = [batch_size] + self._level_lengths[:-1]  # Exclude the final, core decoder length.
        if tensor_rank == 3:
            hierarchy_shape += [-1] + tensor_shape[2:]
        elif tensor_rank != 2:
            # We only expect rank-2 for lengths and rank-3 for sequences.
            raise ValueError(f"Unexpected shape for tensor: {tensor}")
        hierarchy_tensor = k_ops.reshape(tensor, hierarchy_shape)
        # Move the batch dimension to after the hierarchical dimensions.
        perm = list(range(len(hierarchy_shape)))
        perm.insert(self._num_levels, perm.pop(0))
        return k_ops.transpose(hierarchy_tensor, perm)
