import os
import unittest
from pathlib import Path

import keras
from deepdiff import DeepDiff

import resolv_ml.models.autoregressive.rnn.encoders as rnn_encoders


class TestRNNEncoder(unittest.TestCase):

    @property
    def config(self):
        return {
            "output_dir": Path("./output/models/seq2seq/rnn/encoders"),
            "batch_size": 8,
            "seq_length": 64,
            "seq_features": 1,
            "vocabulary_size": 130,
            "embedding_size": 70,
            "enc_rnn_sizes": [16, 64],
            "dropout": 0.0
        }

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)

    def get_model(self):
        inputs = keras.layers.Input(
            shape=(self.config["seq_length"], self.config["seq_features"]),
            batch_size=self.config["batch_size"]
        )
        outputs = rnn_encoders.RNNEncoder(
            enc_rnn_sizes=self.config["enc_rnn_sizes"],
            embedding_layer=keras.layers.Embedding(self.config["vocabulary_size"], self.config["embedding_size"]),
            dropout=self.config["dropout"]
        )(inputs)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def get_input_data_sample(self):
        return keras.ops.ones((self.config["batch_size"], self.config["seq_length"], self.config["seq_features"]))

    def test_predict(self):
        expected_out_shape = self.config["batch_size"], self.config["enc_rnn_sizes"][-1]
        model = self.get_model()
        self.assertTrue(model.output.shape == expected_out_shape)
        input_data_sample = self.get_input_data_sample()
        outputs = model.predict(input_data_sample)
        self.assertTrue(outputs.shape == expected_out_shape)

    def test_saving_and_loading(self):
        model = self.get_model()
        model.save(self.config["output_dir"]/"rnn_encoder.keras")
        loaded_model = keras.saving.load_model(self.config["output_dir"]/"rnn_encoder.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


class TestBidirectionalRNNEncoder(unittest.TestCase):

    @property
    def config(self):
        return {
            "output_dir": Path("./output/models/seq2seq/rnn/encoders"),
            "batch_size": 8,
            "seq_length": 64,
            "seq_features": 1,
            "vocabulary_size": 130,
            "embedding_size": 70,
            "enc_rnn_sizes": [16, 64],
            "dropout": 0.0
        }

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)

    def get_model(self):
        inputs = keras.layers.Input(
            shape=(self.config["seq_length"], self.config["seq_features"]),
            batch_size=self.config["batch_size"]
        )
        outputs = rnn_encoders.BidirectionalRNNEncoder(
            enc_rnn_sizes=self.config["enc_rnn_sizes"],
            embedding_layer=keras.layers.Embedding(self.config["vocabulary_size"], self.config["embedding_size"]),
            dropout=self.config["dropout"]
        )(inputs)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def get_input_data_sample(self):
        return keras.ops.ones((self.config["batch_size"], self.config["seq_length"], self.config["seq_features"]))

    def test_predict(self):
        expected_out_shape = self.config["batch_size"], self.config["enc_rnn_sizes"][-1]*2
        model = self.get_model()
        self.assertTrue(model.output.shape == expected_out_shape)
        input_data_sample = self.get_input_data_sample()
        outputs = model.predict(input_data_sample)
        self.assertTrue(outputs.shape == expected_out_shape)

    def test_saving_and_loading(self):
        model = self.get_model()
        model.save(self.config["output_dir"]/"bidirectional_rnn_encoder.keras")
        loaded_model = keras.saving.load_model(self.config["output_dir"]/"bidirectional_rnn_encoder.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(model.get_config(), loaded_model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


if __name__ == '__main__':
    unittest.main()
