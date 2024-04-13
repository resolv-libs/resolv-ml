import functools
import os
import unittest
from pathlib import Path

import keras
import tensorflow as tf
from keras import losses, metrics
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from resolv_ml.models.dlvm.vae.vanilla_vae import StandardVAE
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

    def get_model(self) -> StandardVAE:
        return StandardVAE(
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
            max_beta=1.0,
            beta_rate=0.0,
            free_bits=0.0
        )

    def load_dataset(self) -> tf.data.TFRecordDataset:
        def map_fn(_, seq):
            empty_aux = tf.zeros((1,))
            input_seq = tf.transpose(seq["pitch_seq"])
            target = input_seq
            return (input_seq, empty_aux), target

        representation = PitchSequenceRepresentation(sequence_length=64)
        tfrecord_loader = TFRecordLoader(
            file_pattern=f"{self.input_dir}/*.tfrecord",
            parse_fn=functools.partial(
                representation.parse_example,
                parse_sequence_feature=True
            ),
            map_fn=map_fn,
            batch_size=32,
            batch_drop_reminder=True,
            deterministic=True,
            seed=42
        )
        return tfrecord_loader.load_dataset()

    def test_trainer(self):
        vae_model = self.get_model()
        trainer = Trainer(vae_model, config_file_path=self.input_dir/"trainer_config.json")
        trainer.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[metrics.SparseCategoricalAccuracy(), metrics.SparseTopKCategoricalAccuracy()]
        )
        trainer.train(train_data=self.load_dataset())


if __name__ == '__main__':
    unittest.main()
