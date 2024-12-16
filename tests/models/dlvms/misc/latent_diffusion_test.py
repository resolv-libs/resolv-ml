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

from resolv_ml.models.dlvm.diffusion.ddim import DDIM
from resolv_ml.models.dlvm.misc.latent_diffusion import LatentDiffusion


class Denoiser(keras.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(128)
        self.dense2 = keras.layers.Dense(1, activation="relu")

    def call(self, inputs, training=None, mask=None):
        noisy_input, _, _ = inputs
        a = self.dense1(noisy_input)
        return self.dense2(a)


class LatentDiffusionTest(unittest.TestCase):

    @property
    def config(self):
        return {
            "input_dir": Path("./"),
            "output_dir": Path("./output/models/dlvm/misc/latent_diffusion"),
            "vae_model_path": "output/models/dlvm/vae/ar-vae/seq2seq/ar_pt_reg_trained.keras",
            "batch_size": 32,
            "sequence_length": 64,
            "sequence_features": 1,
            "z_size": 128,
            "timesteps": 1000,
            "eta": 0,
            "sampling_timesteps": 100,
            "timesteps_scheduler_type": "uniform"
        }

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)

    def get_input_shape(self):
        input_seq_shape = self.config["batch_size"], self.config["sequence_length"], self.config["sequence_features"]
        aux_input_shape = self.config["batch_size"], 1
        return input_seq_shape, aux_input_shape

    def get_latent_diffusion_model(self) -> DDIM:
        diffusion_model = DDIM(
            z_shape=(self.config["z_size"],),
            denoiser=Denoiser(),
            timesteps=self.config["timesteps"],
            noise_level_conditioning=True,
            eta=self.config["eta"],
            sampling_timesteps=self.config["sampling_timesteps"],
            timesteps_scheduler_type=self.config["timesteps_scheduler_type"]
        )
        model = LatentDiffusion(vae=self.config["vae_model_path"], diffusion=diffusion_model)
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

    def _test_model(self, latent_diff_model: LatentDiffusion, output_name: str):
        latent_diff_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        )
        logging.info("Testing model training...")
        latent_diff_model.fit(
            self.load_dataset("train_pitchseq"),
            batch_size=self.config["batch_size"],
            epochs=5,
            steps_per_epoch=5
        )
        latent_diff_model.save(self.config["output_dir"] / output_name)
        # TODO - loading VAE with compile=True does not work after training (seems a Keras bug on optimizers)
        loaded_model = keras.saving.load_model(self.config["output_dir"] / output_name, compile=False)
        loaded_model.compile(run_eagerly=True)
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), latent_diff_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)
        logging.info("Testing model inference...")
        num_sequences = self.config["batch_size"]
        predicted_sequences, predicted_noise, z = latent_diff_model.predict(
            x=keras.ops.convert_to_tensor((num_sequences, self.config["sequence_length"],))
        )
        self.assertTrue(predicted_sequences.shape == (num_sequences, self.config["sequence_length"]))

    def test_seq2seq_latent_diff_model(self):
        latent_diff_model = self.get_latent_diffusion_model()
        self._test_model(latent_diff_model, "seq2seq_latent_diff_trained.keras")


if __name__ == '__main__':
    unittest.main()
