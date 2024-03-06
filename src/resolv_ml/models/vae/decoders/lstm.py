# TODO - DOC
from typing import Tuple, List

import keras
import keras.ops as k_ops

from resolv_ml.models.vae.base import VAEDecoder
from resolv_ml.utilities import rnn as rnn_utils


class LSTMDecoder(VAEDecoder):

    def __init__(self,
                 dec_rnn_sizes: List[int],
                 lstm_dropout: float = 0.0,
                 name: str = "vae/lstm_decoder",
                 **kwargs):
        super(LSTMDecoder, self).__init__(name=name, **kwargs)
        self._stacked_rnn_sizes = dec_rnn_sizes
        self._dropout = lstm_dropout

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: List[Tuple[int, ...]], **kwargs):
        input_sequence_shape = input_shape[0]
        batch_size, sequence_length, token_size = input_sequence_shape
        self._initial_state_layer = rnn_utils.InitialRNNCellStateFromEmbedding(layers_sizes=self._stacked_rnn_sizes,
                                                                               name=f"{self.name}/z_to_initial_state")
        self._stacked_lstm_cells = rnn_utils.StackedLSTM(
            layers_sizes=self._stacked_rnn_sizes,
            return_sequences=False,
            return_state=True,
            lstm_dropout=self._dropout,
            name=f"{self.name}/stacked_lstm_cells"
        )
        self._output_projection = keras.layers.Dense(
            units=token_size,
            activation="linear",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"{self.name}/output_projection"
        )

    def _decode(self, input_sequence, embedding, teacher_force_probability, training: bool = False, **kwargs):
        batch_size, sequence_length, token_size = input_sequence.shape
        # Expand embedding dims to allow for concatenation with the decoder input
        embedding = k_ops.expand_dims(embedding, axis=1)
        initial_state = self._initial_state_layer(embedding, training=training)
        decoder_input = k_ops.zeros(shape=(batch_size, 1, token_size))
        output_sequence = []
        for i in range(sequence_length):
            decoder_input = k_ops.concatenate([decoder_input, embedding], axis=-1)
            dec_emb_output, *initial_state = self._stacked_lstm_cells(decoder_input,
                                                                      initial_state=initial_state,
                                                                      training=training)
            note_emb_out = self._output_projection(dec_emb_output, training=training)
            use_teacher_force = keras.random.randint() < teacher_force_probability
            decoder_input = k_ops.expand_dims(input_sequence[:, i, :], axis=1) if use_teacher_force else note_emb_out
            output_sequence.append(note_emb_out)
        return k_ops.concatenate(output_sequence, axis=1)


class HierarchicalLSTMDecoder(VAEDecoder):

    def __init__(self,
                 level_lengths: List[int],
                 core_decoder: VAEDecoder,
                 dec_rnn_sizes: List[int],
                 lstm_dropout: float = 0.0,
                 name="vae/hierarchical_decoder",
                 **kwargs):
        super(HierarchicalLSTMDecoder, self).__init__(name=name, **kwargs)
        self._level_lengths = level_lengths
        self._num_levels = len(level_lengths) - 1  # subtract 1 for the core decoder level
        self._core_decoder = core_decoder
        self._stacked_rnn_sizes = dec_rnn_sizes
        self._dropout = lstm_dropout
        self._hierarchical_initial_states: List[rnn_utils.InitialRNNCellStateFromEmbedding] = []
        self._stacked_hierarchical_lstm_cells: List[rnn_utils.StackedLSTM] = []

    # noinspection PyAttributeOutsideInit
    def build(self, input_shapes: List[Tuple[int, ...]], **kwargs):
        # Check if given hierarchical level lengths are compatible with input sequence.
        input_sequence_shape = input_shapes[0]
        input_sequence_length = input_sequence_shape[1]
        level_lengths_prod = k_ops.prod(self._level_lengths)
        if input_sequence_length != level_lengths_prod:
            raise ValueError(f"The product of the HierarchicalLSTMDecoder level lengths {level_lengths_prod} must "
                             f"equal the input sequence length {input_sequence_length}.")
        # Build hierarchical layers.
        for level_idx in range(self._num_levels):
            self._hierarchical_initial_states = rnn_utils.InitialRNNCellStateFromEmbedding(
                layers_sizes=self._stacked_rnn_sizes,
                name=f"{self.name}/level_{level_idx}_emb_to_initial_state"
            )
            self._stacked_hierarchical_lstm_cells.append(
                rnn_utils.StackedLSTM(
                    layers_sizes=self._stacked_rnn_sizes,
                    return_sequences=False,
                    return_state=True,
                    lstm_dropout=self._dropout,
                    name=f"{self.name}/level_{level_idx}_lstm"
                )
            )
        # Build core decoder
        self._core_decoder.build(input_shapes)

    def _decode(self, input_sequence, z, teacher_force_probability, training: bool = False, **kwargs):

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
                output, *initial_state = self._stacked_hierarchical_lstm_cells[level](level_input,
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
