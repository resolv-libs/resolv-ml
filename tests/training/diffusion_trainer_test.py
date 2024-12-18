import functools
import logging
import os
import unittest
from pathlib import Path

import keras
import tensorflow as tf
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from resolv_ml.models.dlvm.diffusion.ddim import DDIM
from resolv_ml.models.dlvm.misc.latent_diffusion import LatentDiffusion
from resolv_ml.models.nn.denoisers import DenseDenoiser
from resolv_ml.training.callbacks import LearningRateLoggerCallback
from resolv_ml.training.trainer import Trainer


class TestTrainer(unittest.TestCase):

    @property
    def input_dir(self):
        return Path("./")

    @property
    def output_dir(self):
        return Path("./output/trainer/diffusion")

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_tf_gpu_availability(self):
        gpu_list = tf.config.list_physical_devices('GPU')
        if len(gpu_list) > 0:
            logging.info(f'Num GPUs Available: {len(gpu_list)}. List: {gpu_list}')
        return gpu_list

    def get_input_shape(self):
        input_seq_shape = 32, 64, 1
        aux_input_shape = (32,)
        return input_seq_shape, aux_input_shape

    def get_diffusion_model(self) -> LatentDiffusion:
        diffusion_model = DDIM(
            z_shape=(128,),
            denoiser=DenseDenoiser(),
            timesteps=1000,
            noise_level_conditioning=True,
            eta=0,
            sampling_timesteps=100
        )
        model = LatentDiffusion(
            vae="./output/models/dlvm/vae/ar-vae/seq2seq/ar_pt_reg_trained.keras",
            diffusion=diffusion_model
        )
        model.build(self.get_input_shape())
        return model

    def load_dataset(self, name: str) -> tf.data.TFRecordDataset:
        def map_fn(ctx, seq):
            input_seq = tf.transpose(seq["pitch_seq"])
            attributes = tf.expand_dims(ctx["toussaint"], axis=-1)
            target = input_seq
            return (input_seq, attributes), target

        representation = PitchSequenceRepresentation(sequence_length=64)
        tfrecord_loader = TFRecordLoader(
            file_pattern=f"{self.input_dir}/data/4bars_melodies/{name}.tfrecord",
            parse_fn=functools.partial(
                representation.parse_example,
                parse_sequence_feature=True,
                attributes_to_parse=["contour", "toussaint"]
            ),
            map_fn=map_fn,
            batch_size=32,
            batch_drop_reminder=True,
            deterministic=True,
            seed=42
        )
        return tfrecord_loader.load_dataset()

    def test_trainer(self):
        self.check_tf_gpu_availability()
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            diffusion_model = self.get_diffusion_model()
            trainer = Trainer(
                diffusion_model, config_file_path=self.input_dir / "config" / "diffusion_trainer_config.json"
            )
            trainer.compile(
                lr_schedule=keras.optimizers.schedules.ExponentialDecay(
                    **trainer.config["compile"]["optimizer"]["config"]["learning_rate"]
                )
            )
            trainer.train(
                train_data=self.load_dataset("train_pitchseq"),
                validation_data=self.load_dataset("validation_pitchseq"),
                custom_callbacks=[LearningRateLoggerCallback()]
            )

    def test_model_inference(self):
        loaded_model = keras.saving.load_model(self.output_dir / "runs/checkpoints/epoch_01-val_loss_10.45.keras",
                                               compile=False)
        loaded_model.compile(run_eagerly=True)
        logging.info("Testing model inference...")
        num_sequences, sequence_length = (32, 64)
        predicted_sequences, predicted_noise, z = loaded_model.predict(
            x=keras.ops.convert_to_tensor((num_sequences, sequence_length))
        )
        self.assertTrue(predicted_sequences.shape == (32, 64))


if __name__ == '__main__':
    unittest.main()
