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
    :ivar seq_length: The maximum length of the input sequence.
    :type seq_length: int
    :ivar frequency_base: The base of the frequency exponent.
    :type frequency_base: float
    :ivar frequency_scaling: The scaling factor for the frequency exponent.
    :type frequency_scaling: float
    :ivar positional_encoding: A precomputed tensor containing positional encodings.
    :type positional_encoding: keras.Tensor
    """

    def __init__(self,
                 embedding_dim: int,
                 seq_length: int,
                 frequency_base: float = 10000.0,
                 frequency_scaling: float = 1.0,
                 name: str = "sinusoidal_pe",
                 **kwargs):
        super(SinusoidalPositionalEncoding, self).__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.frequency_base = frequency_base
        self.frequency_scaling = frequency_scaling
        self.positional_encoding = self._compute_positional_encoding()

    def _compute_positional_encoding(self):
        position = keras.ops.expand_dims(keras.ops.arange(0, self.seq_length, dtype="float32"), axis=1)
        div_term = keras.ops.exp(
            keras.ops.arange(0, self.embedding_dim, 2, dtype="float32")
            * -(keras.ops.log(self.frequency_base) / self.embedding_dim)
        )
        freq = self.frequency_scaling * position * div_term
        freq = keras.ops.reshape(freq, newshape=[-1])
        even_embedding_dims = keras.ops.convert_to_tensor(
            [[i, j] for i in range(self.seq_length) for j in range(0, self.embedding_dim, 2)], dtype="int32"
        )
        odd_embedding_dims = keras.ops.convert_to_tensor(
            [[i, j] for i in range(self.seq_length) for j in range(1, self.embedding_dim, 2)], dtype="int32"
        )
        p = keras.ops.zeros(shape=(self.seq_length, self.embedding_dim))
        p = keras.ops.scatter_update(p, indices=even_embedding_dims, updates=keras.ops.sin(freq))
        p = keras.ops.scatter_update(p, indices=odd_embedding_dims, updates=keras.ops.cos(freq))
        return p

    def call(self, timesteps, training: bool = False):
        return keras.ops.take(self.positional_encoding, timesteps, axis=0)
