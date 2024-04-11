import os
import unittest
from pathlib import Path

import keras
from deepdiff import DeepDiff

import resolv_ml.models.seq2seq.rnn.layers as rnn_layers


class TestInitialRNNCellStateFromEmbedding(unittest.TestCase):

    @property
    def output_dir(self):
        return Path("./output/models/seq2seq/rnn/layers")

    @property
    def batch_size(self):
        return 32

    @property
    def embedding_size(self):
        return 128

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, cell_state_sizes):
        inputs = keras.layers.Input(shape=(self.embedding_size,), batch_size=self.batch_size)
        output = rnn_layers.InitialRNNCellStateFromEmbedding(cell_state_sizes=cell_state_sizes)(inputs)
        model = keras.models.Model(inputs=inputs, outputs=output)
        return model

    def test_output_shape(self):
        cell_state_sizes = [[16, 16], [16, 16]]
        model = self.get_model(cell_state_sizes)
        output_shapes = [list(tensor.shape) for tensor in model.output]
        self.assertTrue(cell_state_sizes == output_shapes)

    def test_cell_state_sizes_parameter_checks(self):
        cell_state_sizes = []
        self.assertRaises(ValueError, self.get_model, cell_state_sizes)
        cell_state_sizes = [[16, 16], 3]
        self.assertRaises(ValueError, self.get_model, cell_state_sizes)
        cell_state_sizes = [["16", 16], [16, 16]]
        self.assertRaises(ValueError, self.get_model, cell_state_sizes)

    def test_saving_and_loading(self):
        cell_state_sizes = [[16, 16], [16, 16]]
        model = self.get_model(cell_state_sizes)
        model.save(self.output_dir/"initial_state_from_z.keras")
        loaded_model = keras.saving.load_model(self.output_dir/"initial_state_from_z.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


class TestStackedBidirectionalRNN(unittest.TestCase):

    @property
    def output_dir(self):
        return Path("./output/models/seq2seq/rnn/layers")

    @property
    def batch_size(self):
        return 8

    @property
    def sequence_length(self):
        return 64

    @property
    def sequence_features(self):
        return 1

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, layers_sizes, rnn_cell=None, merge_modes="concat", return_sequences=False, return_state=False):
        inputs = keras.layers.Input(shape=(self.sequence_length, self.sequence_features), batch_size=self.batch_size)
        output = rnn_layers.StackedBidirectionalRNN(
            layers_sizes=layers_sizes,
            rnn_cell=rnn_cell,
            merge_modes=merge_modes,
            return_sequences=return_sequences,
            return_state=return_state
        )(inputs)
        if return_state:

            return keras.models.Model(inputs=inputs, outputs=output)
        model = keras.models.Model(inputs=inputs, outputs=output)
        return model

    def test_output_shape(self):
        layers_sizes = [16, 32]
        # Concat merge mode
        model = self.get_model(layers_sizes=layers_sizes, merge_modes="concat")
        self.assertTrue((self.batch_size, layers_sizes[-1]*2) == model.output.shape)
        # None merge mode
        model = self.get_model(layers_sizes=layers_sizes, merge_modes=None)
        self.assertTrue(len(model.output) == 2)
        self.assertTrue((self.batch_size, layers_sizes[-1]*2) == s.shape for s in model.output)

    def test_return_state_output_shape(self):
        layers_sizes = [16, 32]
        # Concat merge mode
        model = self.get_model(layers_sizes=layers_sizes, return_state=True, merge_modes="concat")
        output, states = model.output[0], model.output[1:]
        self.assertTrue((self.batch_size, layers_sizes[-1]*2) == output.shape)
        self.assertTrue(len(states) == 4)
        self.assertTrue((self.batch_size, layers_sizes[-1]) == x.shape for x in states)
        # None merge mode
        model = self.get_model(layers_sizes=layers_sizes, return_state=True, merge_modes=None)
        outputs, states = model.output[:2], model.output[2:]
        self.assertTrue(len(outputs) == 2)
        self.assertTrue((self.batch_size, layers_sizes[-1]) == x.shape for x in outputs)
        self.assertTrue(len(states) == 4)
        self.assertTrue((self.batch_size, layers_sizes[-1]) == x.shape for x in states)

    def test_return_sequences_output_shape(self):
        model = self.get_model(layers_sizes=[16, 16], return_sequences=True)
        self.assertTrue((self.batch_size, self.sequence_length, 16*2) == model.output.shape)

    def test_layers_sizes_parameter_checks(self):
        layers_sizes = []
        self.assertRaises(ValueError, self.get_model, layers_sizes)
        layers_sizes = [[16, 16], 3]
        self.assertRaises(ValueError, self.get_model, layers_sizes)
        layers_sizes = [["16", 16], [16, 16]]
        self.assertRaises(ValueError, self.get_model, layers_sizes)

    def test_saving_and_loading(self):
        layers_sizes = [16, 32]
        model = self.get_model(layers_sizes)
        model.save(self.output_dir/"stacked_bidirectional_rnn.keras")
        loaded_model = keras.saving.load_model(self.output_dir/"stacked_bidirectional_rnn.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


if __name__ == '__main__':
    unittest.main()
