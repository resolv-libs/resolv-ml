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

from resolv_ml.models.dlvm.vae.ar_vae import AttributeRegularizedVAE, DefaultAttributeRegularization
from resolv_ml.models.seq2seq.rnn import encoders, decoders
from resolv_ml.training.trainer import Trainer


class TestTrainer(unittest.TestCase):

    @property
    def input_dir(self):
        return Path("./data")

    @property
    def output_dir(self):
        return Path("./output/trainer/")

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

    def get_hierarchical_model(self, attribute_regularization_layer=DefaultAttributeRegularization()) \
            -> AttributeRegularizedVAE:
        model = AttributeRegularizedVAE(
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
                    embedding_layer=keras.layers.Embedding(130, 70, name="decoder_embedding")
                ),
                dec_rnn_sizes=[16, 16],
            ),
            attribute_regularization_layer=attribute_regularization_layer,
            max_beta=1.0,
            beta_rate=0.0,
            free_bits=0.0
        )
        model.build(self.get_input_shape())
        return model

    def load_dataset(self, name: str) -> tf.data.TFRecordDataset:
        def map_fn(ctx, seq):
            input_seq = tf.transpose(seq["pitch_seq"])
            attributes = ctx["toussaint"]
            target = input_seq
            return (input_seq, attributes), target

        representation = PitchSequenceRepresentation(sequence_length=64)
        tfrecord_loader = TFRecordLoader(
            file_pattern=f"{self.input_dir}/{name}.tfrecord",
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
            vae_model = self.get_hierarchical_model()
            trainer = Trainer(vae_model, config_file_path=self.input_dir / "trainer_config.json")
            trainer.compile(
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics.SparseCategoricalAccuracy(),
                    metrics.SparseTopKCategoricalAccuracy()
                ],
                lr_schedule=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001,
                    decay_steps=10000,
                    decay_rate=0.99
                )
            )
            trainer.train(
                train_data=self.load_dataset("train_pitchseq"),
                validation_data=self.load_dataset("validation_pitchseq")
            )


if __name__ == '__main__':
    unittest.main()
