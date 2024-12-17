import functools
import logging
import os
import unittest
from pathlib import Path

import keras
import tensorflow as tf
from deepdiff import DeepDiff
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from resolv_ml.models.dlvm.vae.vq_vae import VQVAE
from resolv_ml.models.autoregressive.rnn import encoders, decoders
from resolv_ml.utilities.schedulers import get_scheduler


class Seq2SeqVQVAETest(unittest.TestCase):

    @property
    def config(self):
        return {
            "input_dir": Path("./"),
            "output_dir": Path("./output/models/dlvm/vae/vq-vae/seq2seq"),
            "batch_size": 32,
            "sequence_length": 64,
            "sequence_features": 1,
            "vocabulary_size": 130,
            "embedding_size": 70,
            "enc_rnn_sizes": [16, 16],
            "dec_rnn_sizes": [16, 16],
            "level_lengths": [4, 4, 4],
            "dropout": 0.0,
            "z_size": 32,
            "codebook_size": 64,
            "sampling_scheduler": {
                "type": "constant",
                "config": {
                    "value": 0.5
                }
            }
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

    def get_autoregressive_model(self) -> VQVAE:
        model = VQVAE(
            z_size=self.config["z_size"],
            codebook_size=self.config["codebook_size"],
            feature_extraction_layer=encoders.BidirectionalRNNEncoder(
                enc_rnn_sizes=self.config["enc_rnn_sizes"],
                embedding_layer=self.get_embedding_layer("encoder_embedding"),
                dropout=self.config["dropout"]
            ),
            generative_layer=decoders.RNNAutoregressiveDecoder(
                dec_rnn_sizes=self.config["dec_rnn_sizes"],
                num_classes=self.config["vocabulary_size"],
                embedding_layer=self.get_embedding_layer("decoder_embedding"),
                dropout=self.config["dropout"],
                sampling_scheduler=get_scheduler(
                    schedule_type=self.config["sampling_scheduler"]["type"],
                    schedule_config=self.config["sampling_scheduler"]["config"]
                )
            )
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
            file_pattern=f"{self.config['input_dir']}/data/4bars_melodies/{name}.tfrecord",
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

    def _test_model(self, vae_model: VQVAE, output_name: str):
        vae_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            run_eagerly=True
        )
        logging.info("Testing model training...")
        vae_model.fit(
            self.load_dataset("train_pitchseq"),
            batch_size=self.config["batch_size"],
            epochs=1,
            steps_per_epoch=5
        )
        vae_model.save(self.config["output_dir"] / output_name)
        # TODO - loading VAE with compile=True does not work after training (seems a Keras bug on optimizers)
        loaded_model = keras.saving.load_model(self.config["output_dir"] / output_name, compile=False)
        loaded_model.compile(run_eagerly=True)
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), vae_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)
        logging.info("Testing model inference...")
        num_sequences, sequence_length = (self.config["batch_size"], self.config["sequence_length"])
        predicted_sequences, _ = loaded_model.predict(
            x=keras.ops.convert_to_tensor((num_sequences, sequence_length))
        )
        self.assertTrue(predicted_sequences.shape == (num_sequences, sequence_length))
        logging.info("Testing model inference with encoding...")
        test_sequences = self.load_dataset("test_pitchseq")
        predicted_sequences, latent_codes, _, _, _ = loaded_model.predict(x=test_sequences,
                                                                          batch_size=self.config["batch_size"])
        self.assertTrue(predicted_sequences.shape[-1] == self.config["sequence_length"])
        self.assertTrue(latent_codes.shape[-1] == self.config["z_size"])
        logging.info("Testing model sampling...")
        latent_codes = loaded_model.sample(num_samples=keras.ops.convert_to_tensor(1000))
        self.assertTrue(latent_codes.shape == (1000, self.config["z_size"]))

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

    def test_ar_seq2seq_vae_model(self):
        vae_model = self.get_autoregressive_model()
        self._test_model(vae_model, "ar_seq2seq_vae_trained.keras")


if __name__ == '__main__':
    unittest.main()