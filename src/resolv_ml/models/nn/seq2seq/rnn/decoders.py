# TODO - DOC
import random
from typing import List, Any, Callable

import keras
from keras import ops as k_ops

from . import layers as rnn_layers
from ..base import SequenceDecoder
from .....utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="SequenceDecoders", name="RNNAutoregressiveDecoder")
class RNNAutoregressiveDecoder(SequenceDecoder):

    def __init__(self,
                 dec_rnn_sizes: List[int],
                 num_classes: int,
                 rnn_cell: Any = None,
                 output_projection_layer: keras.Layer = None,
                 embedding_layer: keras.Layer = None,
                 dropout: float = 0.0,
                 sampling_scheduler: Scheduler = None,
                 name: str = "autoregressive_decoder",
                 **kwargs):
        super(RNNAutoregressiveDecoder, self).__init__(
            embedding_layer=embedding_layer,
            sampling_scheduler=sampling_scheduler,
            name=name,
            **kwargs
        )
        self._dec_rnn_sizes = dec_rnn_sizes
        self._num_classes = num_classes
        self._rnn_cell = rnn_cell
        self._output_projection = output_projection_layer
        self._dropout = dropout
        self._stacked_rnn_cells = keras.layers.StackedRNNCells(
            rnn_cell if rnn_cell else [
                rnn_layers.get_default_rnn_cell(size, dropout, name=f"lstm_{cell_idx}")
                for cell_idx, size in enumerate(dec_rnn_sizes)
            ]
        )
        self._initial_state_layer = rnn_layers.InitialRNNCellStateFromEmbedding(
            cell_state_sizes=self._stacked_rnn_cells.state_size,
            name="z_to_initial_state"
        )

    def build(self, input_shape):
        super().build(input_shape)
        input_sequence_shape, aux_input_shape, z_shape = input_shape
        batch_size, _, sequence_features = input_sequence_shape
        self._output_depth = self._num_classes
        if self._embedding_layer:
            if isinstance(self._embedding_layer, keras.layers.Embedding) and sequence_features > 1:
                raise ValueError(f"Can't use an Embedding layer if the sequence has more than one feature."
                                 f"Please use a dense layer instead.")
            embedding_layer_input_shape = (batch_size, sequence_features)
            if not self._embedding_layer.built:
                self._embedding_layer.build(embedding_layer_input_shape)
            self._output_depth = self._embedding_layer.compute_output_shape(embedding_layer_input_shape)[-1]
        if not self._initial_state_layer.built:
            self._initial_state_layer.build(z_shape)
        rnn_input_shape = batch_size, self._output_depth + z_shape[-1]
        self._stacked_rnn_cells.build(rnn_input_shape)
        output_projection_input_shape = batch_size, self._stacked_rnn_cells.output_size
        if self._output_projection:
            projection_layer_out_shape = self._output_projection.compute_output_shape(output_projection_input_shape)
            if projection_layer_out_shape != (batch_size, self._num_classes):
                raise ValueError(f"Given projection layer output shape {projection_layer_out_shape} is not compatible "
                                 f"with the decoder. Output shape must be (batch_size, {self._num_classes}).")
        else:
            self._output_projection = keras.layers.Dense(
                units=self._num_classes,
                activation="linear",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="output_projection"
            )
        if not self._output_projection.built:
            self._output_projection.build(output_projection_input_shape)

    def decode(self, input_sequence, aux_inputs, z, sampling_probability: float = 1.0, **kwargs):
        batch_size, sequence_length, features_length = input_sequence.shape
        initial_state = self._initial_state_layer(z, training=True)
        decoder_input = k_ops.zeros(shape=(batch_size, self._output_depth))
        output_sequence_logits = []
        for i in range(sequence_length):
            predicted_token, output_logits, initial_state = self._predict_sequence_token(
                rnn_input=k_ops.concatenate([decoder_input, z], axis=-1),
                initial_state=initial_state,
                sampling_mode="argmax",
                training=True,
                **kwargs
            )
            output_sequence_logits.append(output_logits)
            teacher_forcing_token = k_ops.squeeze(input_sequence[:, i, :])
            ar_token = teacher_forcing_token if random.random() > sampling_probability else predicted_token
            decoder_input = self._embedding_layer(ar_token)
        return k_ops.stack(output_sequence_logits, axis=1)

    def sample(self, z, seq_length, sampling_mode: str = "argmax", temperature: float = 1.0, **kwargs):
        batch_size = z.shape[0]
        initial_state = self._initial_state_layer(z, training=False)
        decoder_input = k_ops.zeros(shape=(batch_size, self._output_depth))
        predicted_tokens = []
        for i in range(seq_length):
            predicted_token, _, initial_state = self._predict_sequence_token(
                rnn_input=k_ops.concatenate([decoder_input, z], axis=-1),
                initial_state=initial_state,
                sampling_mode=sampling_mode,
                temperature=temperature,
                training=False,
                **kwargs
            )
            predicted_tokens.append(predicted_token)
            decoder_input = self._embedding_layer(predicted_token)
        return k_ops.stack(predicted_tokens, axis=1)

    def _predict_sequence_token(self,
                                rnn_input,
                                initial_state=None,
                                temperature: float = 1.0,
                                sampling_mode: str = "argmax",
                                training: bool = False,
                                seed: int = None):
        dec_emb_output, output_state = self._stacked_rnn_cells(rnn_input, states=initial_state, training=training)
        output_logits = self._output_projection(dec_emb_output, training=training)
        output_logits = output_logits / temperature
        token_probabilities = keras.ops.softmax(output_logits)
        if sampling_mode == "argmax":
            predicted_token = keras.ops.argmax(token_probabilities, axis=-1)
        elif sampling_mode == "categorical":
            predicted_token = keras.random.categorical(output_logits, 1, seed=seed)
        else:
            raise NotImplementedError(f"Sampling mode {sampling_mode} is not implemented.")
        return predicted_token, output_logits, output_state

    def compute_output_shape(self, input_shape):
        input_sequence_shape, aux_input_shape, z_shape = input_shape
        batch_size = input_sequence_shape[0]
        return batch_size, None, None

    def get_config(self):
        base_config = super().get_config()
        config = {
            "dec_rnn_sizes": self._dec_rnn_sizes,
            "num_classes": self._num_classes,
            "rnn_cell": keras.saving.serialize_keras_object(self._rnn_cell),
            "output_projection": keras.saving.serialize_keras_object(self._output_projection),
            "dropout": self._dropout
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        rnn_cell = keras.saving.deserialize_keras_object(config.pop("rnn_cell"))
        output_projection = keras.layers.deserialize(config.pop("output_projection"))
        embedding_layer = keras.layers.deserialize(config.pop("embedding_layer"))
        return cls(rnn_cell=rnn_cell,
                   output_projection_layer=output_projection,
                   embedding_layer=embedding_layer,
                   **config)


@keras.saving.register_keras_serializable(package="SequenceDecoders", name="HierarchicalRNNDecoder")
class HierarchicalRNNDecoder(SequenceDecoder):

    def __init__(self,
                 level_lengths: List[int],
                 core_decoder: SequenceDecoder,
                 dec_rnn_sizes: List[int],
                 rnn_cell: Any = None,
                 dropout: float = 0.0,
                 sampling_scheduler: Scheduler = None,
                 name="hierarchical_decoder",
                 **kwargs):
        super(HierarchicalRNNDecoder, self).__init__(
            sampling_scheduler=sampling_scheduler,
            name=name,
            **kwargs
        )
        self._level_lengths = level_lengths
        self._core_decoder = core_decoder
        self._dec_rnn_sizes = dec_rnn_sizes
        self._rnn_cell = rnn_cell
        self._dropout = dropout
        self._num_levels = len(level_lengths) - 1  # subtract 1 for the core decoder level
        self._hierarchical_initial_states: List[rnn_layers.InitialRNNCellStateFromEmbedding] = []
        self._stacked_hierarchical_rnn_cells: List[keras.layers.StackedRNNCells] = []
        for level_idx in range(self._num_levels):
            level_rnn_cells = keras.layers.StackedRNNCells(
                rnn_cell if rnn_cell else [
                    rnn_layers.get_default_rnn_cell(size, dropout, name=f"lstm_{level_idx}_{cell_idx}")
                    for cell_idx, size in enumerate(dec_rnn_sizes)
                ],
                name=f"level_{level_idx}_stacked_rnn_cells"
            )
            self._hierarchical_initial_states.append(
                rnn_layers.InitialRNNCellStateFromEmbedding(
                    cell_state_sizes=level_rnn_cells.state_size,
                    name=f"level_{level_idx}_emb_to_initial_state"
                )
            )
            self._stacked_hierarchical_rnn_cells.append(level_rnn_cells)

    def build(self, input_shape):
        super().build(input_shape)
        input_sequence_shape, aux_input_shape, z_shape = input_shape
        batch_size, sequence_length, sequence_features = input_sequence_shape
        # Check if given hierarchical level lengths are compatible with input sequence.
        level_lengths_prod = k_ops.prod(self._level_lengths)
        if sequence_length != level_lengths_prod:
            raise ValueError(f"The product of the HierarchicalLSTMDecoder level lengths {level_lengths_prod} must "
                             f"equal the input sequence length {sequence_length}.")
        rnn_cells_input_shape = batch_size, 1
        initial_states_input_shape = z_shape
        for level_idx in range(self._num_levels):
            self._hierarchical_initial_states[level_idx].build(initial_states_input_shape)
            current_level_rnn_cells = self._stacked_hierarchical_rnn_cells[level_idx]
            current_level_rnn_cells.build(rnn_cells_input_shape)
            initial_states_input_shape = batch_size, current_level_rnn_cells.output_size
        input_sequence_hier_shape = self._get_hierarchy_shape(input_sequence_shape)
        core_decoder_input_seq_shape = input_sequence_hier_shape[self._num_levels:]
        core_decoder_aux_input_shape = aux_input_shape
        core_decoder_z_shape = batch_size, self._stacked_hierarchical_rnn_cells[-1].output_size
        self._core_decoder.build((core_decoder_input_seq_shape, core_decoder_aux_input_shape, core_decoder_z_shape))

    def decode(self, input_sequence, aux_inputs, z, sampling_probability: float = 1.0, **kwargs):

        def base_decode(embedding, path: List[int] = None):
            base_input_sequence = hierarchy_input_sequence[path]
            return self._core_decoder.decode(
                input_sequence=base_input_sequence,
                aux_inputs=aux_inputs,
                z=embedding,
                sampling_probability=sampling_probability,
                **kwargs
            )

        hierarchy_input_sequence = self._reshape_to_hierarchy(input_sequence)
        output_sequence = self._hierarchical_decode(z, base_fn=base_decode, training=True)
        return k_ops.concatenate(output_sequence, axis=1)

    def sample(self, z, seq_length, sampling_mode: str = "argmax", temperature: float = 1.0, **kwargs):
        # TODO - add support for custom sequence lengths

        def base_sample(embedding, _):
            return self._core_decoder.sample(
                z=embedding,
                seq_length=self._level_lengths[-1],
                sampling_mode=sampling_mode,
                temperature=temperature,
                **kwargs
            )

        output_sequence = self._hierarchical_decode(z, base_fn=base_sample, training=False)
        return k_ops.concatenate(output_sequence, axis=1)

    def _hierarchical_decode(self, z, base_fn: Callable, training: bool = False):

        def recursive_decode(embedding, path: List[int] = None):
            """Recursive hierarchical decode function."""
            path = path or []
            level = len(path)
            if level == self._num_levels:
                decoded_subsequence = base_fn(embedding, path)
                decoded_sequence.append(decoded_subsequence)
                return decoded_subsequence
            initial_state = self._hierarchical_initial_states[level](embedding, training=training)
            for idx in range(self._level_lengths[level]):
                level_input = k_ops.zeros(shape=(batch_size, 1))
                output, initial_state = self._stacked_hierarchical_rnn_cells[level](level_input,
                                                                                    states=initial_state,
                                                                                    training=training)
                recursive_decode(output, path + [idx])

        batch_size = z.shape[0]
        decoded_sequence = []
        # This will populate decoded_sequence with all the decoded subsequences
        recursive_decode(z)
        return decoded_sequence

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
            raise ValueError(f"Unexpected shape: {tensor_shape}")
        hierarchy_tensor = k_ops.reshape(tensor, hierarchy_shape)
        # Move the batch dimension to after the hierarchical dimensions.
        perm = list(range(len(hierarchy_shape)))
        perm.insert(self._num_levels, perm.pop(0))
        return k_ops.transpose(hierarchy_tensor, perm)

    def _get_hierarchy_shape(self, tensor_shape):
        tensor = k_ops.zeros(tensor_shape)
        hier_tensor = self._reshape_to_hierarchy(tensor)
        return hier_tensor.shape

    def get_config(self):
        base_config = super().get_config()
        base_config.pop("embedding_layer")
        config = {
            "level_lengths": self._level_lengths,
            "core_decoder": keras.saving.serialize_keras_object(self._core_decoder),
            "dec_rnn_sizes": self._dec_rnn_sizes,
            "rnn_cell": keras.saving.serialize_keras_object(self._rnn_cell),
            "dropout": self._dropout
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        rnn_cell = keras.saving.deserialize_keras_object(config.pop("rnn_cell"))
        core_decoder = keras.saving.deserialize_keras_object(config.pop("core_decoder"))
        return cls(rnn_cell=rnn_cell, core_decoder=core_decoder, **config)
