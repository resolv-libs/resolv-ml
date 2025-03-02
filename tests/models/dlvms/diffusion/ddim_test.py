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
from resolv_ml.models.nn.denoisers import DenseDenoiser
from resolv_ml.utilities.schedulers import ConstantScheduler


class DDIMTest(unittest.TestCase):

    @property
    def config(self):
        return {
            "input_dir": Path("./"),
            "output_dir": Path("./output/models/dlvm/diffusion/ddim"),
            "batch_size": 32,
            "sequence_length": 64,
            "sequence_features": 1,
            "z_size": 64,
            "timesteps": 1000,
            "eta": 5.,
            "sampling_timesteps": 10,
            "timesteps_scheduler_type": "uniform",
            "cfg_uncond_probability_scheduler": ConstantScheduler(0.2),
            "cfg_weight": 3.0
        }

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)

    def get_input_shape(self):
        input_seq_shape = self.config["batch_size"], self.config["sequence_length"], self.config["sequence_features"]
        aux_input_shape = self.config["batch_size"], 1
        return input_seq_shape, aux_input_shape

    def get_ddim_model(self) -> DDIM:

        model = DDIM(
            z_shape=(self.config["z_size"], self.config["sequence_features"]),
            denoiser=DenseDenoiser(),
            timesteps=self.config["timesteps"],
            noise_level_conditioning=True,
            eta=self.config["eta"],
            sampling_timesteps=self.config["sampling_timesteps"],
            timesteps_scheduler_type=self.config["timesteps_scheduler_type"],
            cfg_uncond_probability_scheduler=self.config["cfg_uncond_probability_scheduler"],
            cfg_weight=self.config["cfg_weight"]
        )
        model.build(self.get_input_shape())
        return model

    def load_dataset(self, name: str) -> tf.data.TFRecordDataset:

        def map_fn(ctx, seq):
            input_seq = tf.transpose(seq["pitch_seq"])
            attributes = tf.expand_dims(ctx["contour"], axis=-1)
            target = input_seq
            return (input_seq, attributes), target

        representation = PitchSequenceRepresentation(sequence_length=self.config["sequence_length"])
        tfrecord_loader = TFRecordLoader(
            file_pattern=f"{self.config['input_dir']}/data/4bars_melodies/{name}.tfrecord",
            parse_fn=functools.partial(
                representation.parse_example,
                parse_sequence_feature=True,
                attributes_to_parse=["contour"]
            ),
            map_fn=map_fn,
            batch_size=self.config["batch_size"],
            batch_drop_reminder=True,
            deterministic=True,
            seed=42
        )
        return tfrecord_loader.load_dataset()

    def _test_model(self, ddim_model: DDIM, output_name: str):
        ddim_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        )
        logging.info("Testing model training...")
        ddim_model.fit(
            self.load_dataset("train_pitchseq"),
            batch_size=self.config["batch_size"],
            epochs=2,
            steps_per_epoch=5
        )
        ddim_model.save(self.config["output_dir"] / output_name)
        # TODO - loading VAE with compile=True does not work after training (seems a Keras bug on optimizers)
        loaded_model = keras.saving.load_model(self.config["output_dir"] / output_name, compile=False)
        loaded_model.compile(run_eagerly=True)
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), ddim_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)
        logging.info("Testing model inference...")
        num_sequences = self.config["batch_size"]
        labels = keras.ops.expand_dims(keras.ops.linspace(0, 10, num_sequences), axis=-1)
        predicted_sequences, predicted_noise, z = ddim_model.predict(
            x=(keras.ops.full((num_sequences,), num_sequences, dtype="int32"), labels)
        )
        self.assertTrue(predicted_sequences.shape == (num_sequences, self.config["sampling_timesteps"],
                                                      self.config["sequence_length"], self.config["sequence_features"]))

    def test_seq2seq_ddim_model(self):
        ddim_model = self.get_ddim_model()
        self._test_model(ddim_model, "seq2seq_ddim_trained.keras")


if __name__ == '__main__':
    unittest.main()
