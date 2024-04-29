import functools
import os
import unittest
from pathlib import Path

import keras
import tensorflow as tf
from deepdiff import DeepDiff
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from resolv_ml.models.dlvm.vae.vanilla_vae import StandardVAE
from resolv_ml.models.seq2seq.rnn import encoders, decoders


class Seq2SeqStandardVAETest(unittest.TestCase):

    @property
    def config(self):
        return {
            "input_dir": Path("./data"),
            "output_dir": Path("./output/models/dlvm/vae/standard-vae/seq2seq"),
            "batch_size": 32,
            "sequence_length": 64,
            "sequence_features": 1,
            "vocabulary_size": 130,
            "embedding_size": 70,
            "enc_rnn_sizes": [16, 16],
            "dec_rnn_sizes": [16, 16],
            "level_lengths": [4, 4, 4],
            "dropout": 0.0,
            "z_size": 128,
            "sampling_schedule": "constant",
            "sampling_rate": 0.0,
            "max_beta": 1.0,
            "beta_rate": 0.0,
            "free_bits": 0.0
        }

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)

    def get_embedding_layer(self, name: str) -> keras.layers.Embedding:
        return keras.layers.Embedding(self.config["vocabulary_size"], self.config["embedding_size"], name=name)

    def get_input_shape(self):
        input_seq_shape = self.config["batch_size"], self.config["sequence_length"], self.config["sequence_features"]
        aux_input_shape = self.config["batch_size"], 1
        return input_seq_shape, aux_input_shape

    def get_autoregressive_model(self) -> StandardVAE:
        model = StandardVAE(
            z_size=self.config["z_size"],
            input_processing_layer=encoders.BidirectionalRNNEncoder(
                enc_rnn_sizes=self.config["enc_rnn_sizes"],
                embedding_layer=self.get_embedding_layer("encoder_embedding"),
                dropout=self.config["dropout"]
            ),
            generative_layer=decoders.RNNAutoregressiveDecoder(
                dec_rnn_sizes=self.config["dec_rnn_sizes"],
                num_classes=self.config["vocabulary_size"],
                embedding_layer=self.get_embedding_layer("decoder_embedding"),
                dropout=self.config["dropout"],
                sampling_schedule=self.config["sampling_schedule"],
                sampling_rate=self.config["sampling_rate"]
            ),
            max_beta=self.config["max_beta"],
            beta_rate=self.config["beta_rate"],
            free_bits=self.config["free_bits"]
        )
        model.build(self.get_input_shape())
        return model

    def get_hierarchical_model(self) -> StandardVAE:
        model = StandardVAE(
            z_size=self.config["z_size"],
            input_processing_layer=encoders.BidirectionalRNNEncoder(
                enc_rnn_sizes=self.config["enc_rnn_sizes"],
                embedding_layer=self.get_embedding_layer("encoder_embedding"),
                dropout=self.config["dropout"]
            ),
            generative_layer=decoders.HierarchicalRNNDecoder(
                level_lengths=self.config["level_lengths"],
                core_decoder=decoders.RNNAutoregressiveDecoder(
                    dec_rnn_sizes=self.config["dec_rnn_sizes"],
                    num_classes=self.config["vocabulary_size"],
                    embedding_layer=self.get_embedding_layer("decoder_embedding"),
                    dropout=self.config["dropout"],
                    sampling_schedule=self.config["sampling_schedule"],
                    sampling_rate=self.config["sampling_rate"]
                ),
                dec_rnn_sizes=self.config["dec_rnn_sizes"],
                dropout=self.config["dropout"]
            ),
            max_beta=self.config["max_beta"],
            beta_rate=self.config["beta_rate"],
            free_bits=self.config["free_bits"]
        )
        model.build(self.get_input_shape())
        return model

    def load_dataset(self, name: str) -> tf.data.TFRecordDataset:
        def map_fn(_, seq):
            empty_aux = tf.zeros(shape=(self.config["batch_size"], 1))
            input_seq = tf.transpose(seq["pitch_seq"])
            target = input_seq
            return (input_seq, empty_aux), target

        representation = PitchSequenceRepresentation(sequence_length=self.config["sequence_length"])
        tfrecord_loader = TFRecordLoader(
            file_pattern=f"{self.config['input_dir']}/{name}.tfrecord",
            parse_fn=functools.partial(
                representation.parse_example,
                parse_sequence_feature=True
            ),
            map_fn=map_fn,
            batch_size=self.config["batch_size"],
            batch_drop_reminder=True,
            deterministic=True,
            seed=42
        )
        return tfrecord_loader.load_dataset()

    def test_ar_seq2seq_vae_summary_and_plot(self):
        vae_model = self.get_autoregressive_model()
        vae_model.print_summary(self.get_input_shape(), expand_nested=True)
        keras.utils.plot_model(
            vae_model.build_graph(self.get_input_shape()),
            show_shapes=True,
            show_layer_names=True,
            show_dtype=True,
            to_file=self.config["output_dir"] / "seq2seq_vae_plot.png",
            expand_nested=True
        )
        self.assertTrue(vae_model)

    def test_ar_seq2seq_vae_save_and_loading(self):
        vae_model = self.get_autoregressive_model()
        vae_model.save(self.config["output_dir"] / "ar_seq2seq_vae.keras")
        loaded_model = keras.saving.load_model(self.config["output_dir"] / "ar_seq2seq_vae.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), vae_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)

    def test_ar_seq2seq_vae_training(self):
        vae_model = self.get_autoregressive_model()
        vae_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy()],
            run_eagerly=True
        )
        vae_model.fit(
            self.load_dataset("train_pitchseq"),
            validation_data=self.load_dataset("validation_pitchseq"),
            batch_size=self.config["batch_size"],
            epochs=1,
            steps_per_epoch=5
        )
        vae_model.save(self.config["output_dir"] / "ar_seq2seq_vae_trained.keras")
        # TODO - Can't compile the model again after loading don't know why
        loaded_model = keras.saving.load_model(self.config["output_dir"] / "ar_seq2seq_vae_trained.keras",
                                               compile=False)
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), vae_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)

    def test_hier_seq2seq_vae_save_and_loading(self):
        vae_model = self.get_autoregressive_model()
        vae_model.save(self.config["output_dir"] / "hier_seq2seq_vae.keras")
        loaded_model = keras.saving.load_model(self.config["output_dir"] / "hier_seq2seq_vae.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), vae_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)

    def test_hier_seq2seq_vae_training(self):
        vae_model = self.get_hierarchical_model()
        vae_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            run_eagerly=True
        )
        vae_model.fit(
            self.load_dataset("train_pitchseq"),
            validation_data=self.load_dataset("validation_pitchseq"),
            batch_size=self.config["batch_size"],
            epochs=1,
            steps_per_epoch=5
        )
        vae_model.save(self.config["output_dir"] / "hier_seq2seq_vae_trained.keras")
        # TODO - Can't compile the model again after loading don't know why
        loaded_model = keras.saving.load_model(self.config["output_dir"] / "hier_seq2seq_vae_trained.keras",
                                               compile=False)
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), vae_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


if __name__ == '__main__':
    unittest.main()
