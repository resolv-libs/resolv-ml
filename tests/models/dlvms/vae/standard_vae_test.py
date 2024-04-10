import functools
import os
import unittest
from pathlib import Path

import keras
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation
import tensorflow as tf

from resolv_ml.models.dlvm.vae.vanilla_vae import StandardVAE
from resolv_ml.models.seq2seq.rnn.decoders import RNNAutoregressiveDecoder
from resolv_ml.models.seq2seq.rnn.encoders import BidirectionalRNNEncoder


class Seq2SeqVAETest(unittest.TestCase):

    @property
    def output_dir(self) -> Path:
        return Path("./output/models/dlvm/vae")

    @property
    def input_dir(self) -> Path:
        return Path("./data")

    @property
    def batch_size(self) -> int:
        return 32

    @property
    def sequence_length(self) -> int:
        return 64

    @property
    def sequence_features(self) -> int:
        return 1

    @property
    def num_notes(self) -> int:
        return 130

    @property
    def embedding_size(self) -> int:
        return 70

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, print_plot_to_path: str = None) -> StandardVAE:
        return StandardVAE(
            input_shape=(self.sequence_length, self.sequence_features),
            z_size=128,
            input_processing_layer=BidirectionalRNNEncoder(
                num_classes=self.num_notes,
                enc_rnn_sizes=[16, 16],
                embedding_layer=keras.layers.Embedding(input_dim=self.num_notes, output_dim=self.embedding_size,
                                                       name="encoder_embedding"),
            ),
            generative_layer=RNNAutoregressiveDecoder(
                num_classes=self.num_notes,
                dec_rnn_sizes=[16, 16],
                embedding_layer=keras.layers.Embedding(input_dim=self.num_notes, output_dim=self.embedding_size,
                                                       name="decoder_embedding"),
            )
        )

    def load_dataset(self) -> tf.data.TFRecordDataset:
        def map_fn(_, seq):
            empty_aux = tf.zeros(self.batch_size)
            input_seq = tf.transpose(seq["pitch_seq"])
            target = input_seq
            return (input_seq, empty_aux), target

        representation = PitchSequenceRepresentation(sequence_length=self.sequence_length)
        tfrecord_loader = TFRecordLoader(
            file_pattern=f"{self.input_dir}/*.tfrecord",
            parse_fn=functools.partial(
                representation.parse_example,
                parse_sequence_feature=True
            ),
            map_fn=map_fn,
            batch_size=self.batch_size,
            batch_drop_reminder=True,
            deterministic=True,
            seed=42
        )
        return tfrecord_loader.load_dataset()

    def test_seq2seq_vae_summary(self):
        input_shape = self.batch_size, self.sequence_length, self.sequence_features
        vae_model = self.get_model()
        vae_model.build(input_shape)
        vae_model.summary(expand_nested=True)
        vae_model.plot(self.output_dir / "seq2seq.png")
        self.assertTrue(vae_model)

    def test_seq2seq_vae_training(self):
        input_shape = self.batch_size, self.sequence_length, self.sequence_features
        vae_model = self.get_model()
        vae_model.build(input_shape)
        dataset = self.load_dataset()
        vae_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            run_eagerly=True
        )
        vae_model.fit(dataset, batch_size=32, epochs=1)
        self.assertTrue(vae_model)


if __name__ == '__main__':
    unittest.main()
