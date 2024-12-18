import os
import unittest
from pathlib import Path

import keras
from deepdiff import DeepDiff

import resolv_ml.models.seq2seq.rnn.decoders as rnn_decoders


class TestRNNAutoregressiveDecoder(unittest.TestCase):

    @property
    def config(self):
        return {
            "output_dir": Path("./output/models/seq2seq/rnn/decoders"),
            "batch_size": 8,
            "seq_length": 64,
            "seq_features": 1,
            "vocabulary_size": 130,
            "embedding_size": 70,
            "dec_rnn_sizes": [16, 64],
            "z_size": 32,
            "dropout": 0.0
        }

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)

    def get_decoder(self):
        decoder = rnn_decoders.RNNAutoregressiveDecoder(
            dec_rnn_sizes=self.config["dec_rnn_sizes"],
            num_classes=self.config["vocabulary_size"],
            embedding_layer=keras.layers.Embedding(self.config["vocabulary_size"], self.config["embedding_size"]),
            dropout=self.config["dropout"]
        )
        decoder.build(self.get_decoder_input_shape())
        return decoder

    def get_decoder_input_shape(self):
        input_sequence_shape = self.config["batch_size"], self.config["seq_length"], self.config["seq_features"]
        aux_input_shape = (1,)
        z_shape = self.config["batch_size"], self.config["z_size"]
        return input_sequence_shape, aux_input_shape, z_shape

    def get_input_data_sample(self):
        inputs = keras.ops.ones((self.config["batch_size"], self.config["seq_length"], self.config["seq_features"]))
        aux_inputs = keras.ops.ones((self.config["batch_size"], 1))
        z_input = keras.ops.ones((self.config["batch_size"], self.config["z_size"]))
        return inputs, aux_inputs, z_input

    def test_decode(self):
        expected_out_shape = self.config["batch_size"], self.config["seq_length"], self.config["vocabulary_size"]
        input_data_sample = self.get_input_data_sample()
        decoder = self.get_decoder()
        outputs = decoder(input_data_sample, training=True)
        self.assertTrue(outputs.shape == expected_out_shape)

    def test_sampling(self):
        expected_out_shape_decode = (self.config["batch_size"], self.config["seq_length"])
        _, _, z_input = self.get_input_data_sample()
        decoder = self.get_decoder()
        outputs = decoder((z_input, keras.ops.convert_to_tensor(self.config["seq_length"])), training=False)
        self.assertTrue(outputs.shape == expected_out_shape_decode)

    def test_saving_and_loading(self):
        model = self.get_decoder()
        model.save(self.config["output_dir"]/"rnn_decoder.keras")
        loaded_model = keras.saving.load_model(self.config["output_dir"]/"rnn_decoder.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


class TestHierarchicalRNNDecoder(unittest.TestCase):

    @property
    def config(self):
        return {
            "output_dir": Path("./output/models/seq2seq/rnn/decoders"),
            "batch_size": 8,
            "seq_length": 64,
            "seq_features": 1,
            "vocabulary_size": 130,
            "embedding_size": 70,
            "level_lengths": [4, 4, 4],
            "dec_rnn_sizes": [16, 64],
            "z_size": 32,
            "dropout": 0.0
        }

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)

    def get_decoder(self):
        decoder = rnn_decoders.HierarchicalRNNDecoder(
            level_lengths=self.config["level_lengths"],
            core_decoder=rnn_decoders.RNNAutoregressiveDecoder(
                dec_rnn_sizes=self.config["dec_rnn_sizes"],
                num_classes=self.config["vocabulary_size"],
                embedding_layer=keras.layers.Embedding(self.config["vocabulary_size"], self.config["embedding_size"]),
                dropout=self.config["dropout"]
            ),
            dec_rnn_sizes=self.config["dec_rnn_sizes"],
            dropout=self.config["dropout"]
        )
        decoder.build(self.get_decoder_input_shape())
        return decoder

    def get_decoder_input_shape(self):
        input_sequence_shape = self.config["batch_size"], self.config["seq_length"], self.config["seq_features"]
        aux_input_shape = (1,)
        z_shape = self.config["batch_size"], self.config["z_size"]
        return input_sequence_shape, aux_input_shape, z_shape

    def get_input_data_sample(self):
        inputs = keras.ops.ones((self.config["batch_size"], self.config["seq_length"], self.config["seq_features"]))
        aux_inputs = keras.ops.ones((self.config["batch_size"], 1))
        z_input = keras.ops.ones((self.config["batch_size"], self.config["z_size"]))
        return inputs, aux_inputs, z_input

    def test_decode(self):
        expected_out_shape = self.config["batch_size"], self.config["seq_length"], self.config["vocabulary_size"]
        input_data_sample = self.get_input_data_sample()
        decoder = self.get_decoder()
        outputs = decoder(input_data_sample, training=True)
        self.assertTrue(outputs.shape == expected_out_shape)

    def test_sampling(self):
        expected_out_shape_decode = (self.config["batch_size"], self.config["seq_length"])
        _, _, z_input = self.get_input_data_sample()
        decoder = self.get_decoder()
        outputs = decoder((z_input, keras.ops.convert_to_tensor(self.config["seq_length"])), training=False)
        self.assertTrue(outputs.shape == expected_out_shape_decode)

    def test_saving_and_loading(self):
        model = self.get_decoder()
        model.save(self.config["output_dir"]/"hier_rnn_decoder.keras")
        loaded_model = keras.saving.load_model(self.config["output_dir"]/"hier_rnn_decoder.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


if __name__ == '__main__':
    unittest.main()
