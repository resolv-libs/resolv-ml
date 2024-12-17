# TODO - DOC
import keras


class SinusoidalPositionalEncoding(keras.Layer):
    """
    SinusoidalPositionalEncoding is a layer for computing fixed sinusoidal positional encodings.
    As proposed in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) by Vaswani et al 2017.

    This layer generates sinusoidal positional encodings, which are used in transformer to encode
    the positions of tokens in a sequence. The positional encodings are deterministic
    and depend on the sequence length and embedding dimension. They allow the model to leverage
    the relative or absolute position of tokens in the sequence during training. This layer does
    not have trainable parameters.

    The positional encodings are calculated using sine and cosine functions of varying frequencies,
    and are independent of the values of the inputs. These encodings are precomputed during
    initialization and indexed during the forward pass.

    :ivar embedding_dim: The dimension of the embedding representation.
    :type embedding_dim: int
    :ivar max_seq_length: The maximum length of the input sequence.
    :type max_seq_length: int
    :ivar frequency_base: The base of the frequency exponent.
    :type frequency_base: float
    :ivar frequency_scaling: The scaling factor for the frequency exponent.
    :type frequency_scaling: float
    :ivar lookup_table: Whether to precompute the positional encodings.
    :type lookup_table: bool
    :ivar positional_encoding: A precomputed tensor containing positional encodings.
    :type positional_encoding: keras.Tensor
    """

    def __init__(self,
                 embedding_dim: int,
                 max_seq_length: int = 0,
                 frequency_base: float = 10000.0,
                 frequency_scaling: float = 1.0,
                 lookup_table: bool = False,
                 name: str = "sinusoidal_pe",
                 **kwargs):
        super(SinusoidalPositionalEncoding, self).__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.frequency_base = frequency_base
        self.frequency_scaling = frequency_scaling
        self.lookup_table = lookup_table
        if lookup_table:
            if not max_seq_length:
                raise ValueError("max_seq_length must be provided if lookup_table is True.")
            self.positional_encoding = self._compute_positional_encoding(positions=keras.ops.arange(0, max_seq_length))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_dim

    def _compute_positional_encoding(self, positions):
        positions = keras.ops.expand_dims(positions, axis=1) if len(positions.shape) == 1 else positions
        positions = keras.ops.cast(positions, dtype="float32")
        div_term = keras.ops.exp(
            keras.ops.arange(0, self.embedding_dim, 2, dtype="float32")
            * -(keras.ops.log(self.frequency_base) / self.embedding_dim)
        )
        freq = self.frequency_scaling * positions * div_term
        p = keras.ops.concatenate([keras.ops.sin(freq), keras.ops.cos(freq)], axis=-1)
        return p

    def call(self, positions, training: bool = False):
        if self.lookup_table:
            positions = keras.ops.squeeze(positions, axis=-1) if len(positions.shape) > 1 else positions
            positions = keras.ops.cast(positions, dtype="int32")
            return keras.ops.take(self.positional_encoding, positions, axis=0)
        else:
            return self._compute_positional_encoding(positions)
