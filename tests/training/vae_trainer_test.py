import functools
import logging
import os
import unittest
from pathlib import Path

import keras
import tensorflow as tf
from keras import losses, metrics
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from resolv_ml.models.dlvm.vae.vanilla_vae import StandardVAE
from resolv_ml.models.seq2seq.rnn import decoders, encoders
from resolv_ml.training.callbacks import LearningRateLoggerCallback
from resolv_ml.training.trainer import Trainer
from resolv_ml.utilities.schedulers import get_scheduler
from training import utils


class TestVAETrainer(unittest.TestCase):

    @property
    def input_dir(self):
        return Path("./")

    @property
    def output_dir(self):
        return Path("./output/trainer/vae")

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_input_shape(self):
        input_seq_shape = 32, 64, 1
        aux_input_shape = (32,)
        return input_seq_shape, aux_input_shape

    def get_hierarchical_model(self) -> StandardVAE:
        model = StandardVAE(
            z_size=128,
            input_processing_layer=encoders.BidirectionalRNNEncoder(
                enc_rnn_sizes=[16, 16],
                embedding_layer=keras.layers.Embedding(130, 70, name="encoder_embedding")
            ),
            generative_layer=decoders.HierarchicalRNNDecoder(
                level_lengths=[4, 4, 4],
                core_decoder=decoders.RNNAutoregressiveDecoder(
                    dec_rnn_sizes=[16, 16],
                    num_classes=130,
                    embedding_layer=keras.layers.Embedding(130, 70, name="decoder_embedding"),
                    sampling_scheduler=get_scheduler(
                        schedule_type="sigmoid",
                        schedule_config={
                            "rate": 200,
                            "min_value": 0.0,
                            "max_value": 1.0,
                            "decay": False
                        }
                    )
                ),
                dec_rnn_sizes=[16, 16],
            ),
            div_beta_scheduler=get_scheduler(
                schedule_type="exponential",
                schedule_config={
                    "rate": 0.999,
                    "min_value": 0.0,
                    "max_value": 0.3,
                    "decay": False
                }
            ),
            free_bits=0.0
        )
        model.build(self.get_input_shape())
        return model

    def get_flat_model(self) -> StandardVAE:
        model = StandardVAE(
            z_size=128,
            feature_extraction_layer=encoders.BidirectionalRNNEncoder(
                enc_rnn_sizes=[16, 16],
                embedding_layer=keras.layers.Embedding(130, 70, name="encoder_embedding")
            ),
            generative_layer=decoders.RNNAutoregressiveDecoder(
                dec_rnn_sizes=[16, 16],
                num_classes=130,
                embedding_layer=keras.layers.Embedding(130, 70, name="decoder_embedding"),
                sampling_scheduler=get_scheduler(
                    schedule_type="sigmoid",
                    schedule_config={
                        "rate": 200,
                        "min_value": 0.0,
                        "max_value": 1.0,
                        "decay": False
                    }
                )
            ),
            div_beta_scheduler=get_scheduler(
                schedule_type="exponential",
                schedule_config={
                    "rate": 0.999,
                    "min_value": 0.0,
                    "max_value": 0.3,
                    "decay": False
                }
            ),
            free_bits=0.0
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
        strategy = utils.get_distributed_strategy(0, True)
        with strategy.scope():
            vae_model = self.get_hierarchical_model()
            trainer = Trainer(
                vae_model, config_file_path=self.input_dir / "config" / "vae_trainer_config.json"
            )
            trainer.compile(
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics.SparseCategoricalAccuracy(),
                    keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="sparse_top_3_categorical_accuracy"),
                    keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="sparse_top_5_categorical_accuracy")
                ],
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
        loaded_model = keras.saving.load_model(self.output_dir / "runs/checkpoints/epoch_01-val_loss_8.93.keras",
                                               compile=False)
        loaded_model.compile(run_eagerly=True)
        logging.info("Testing model inference...")
        num_sequences, sequence_length = (32, 64)
        predicted_sequences, _ = loaded_model.predict(x=keras.ops.convert_to_tensor((num_sequences, sequence_length)))
        self.assertTrue(predicted_sequences.shape == (32, 64))
        logging.info("Testing model inference with encoding...")
        test_sequences = self.load_dataset("test_pitchseq")
        predicted_sequences, latent_codes, _, _, _ = loaded_model.predict(x=test_sequences, batch_size=32)
        self.assertTrue(predicted_sequences.shape[-1] == 64)
        self.assertTrue(latent_codes.shape[-1] == 128)
        logging.info("Testing model sampling...")
        latent_codes = loaded_model.get_latent_codes(n=keras.ops.convert_to_tensor(1000))
        self.assertTrue(latent_codes.shape == (1000, 128))


if __name__ == '__main__':
    unittest.main()
