import os
import unittest
from pathlib import Path

import keras
from deepdiff import DeepDiff

import resolv_ml.models.nn.seq2seq.rnn.layers as rnn_layers


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

    def get_input_data_sample(self):
        return keras.ops.ones((self.batch_size, self.embedding_size))

    def get_model(self, cell_state_sizes):
        inputs = keras.layers.Input(shape=(self.embedding_size,), batch_size=self.batch_size)
        outputs = rnn_layers.InitialRNNCellStateFromEmbedding(cell_state_sizes=cell_state_sizes)(inputs)
        model = keras.models.Model(inputs=inputs, outputs=[x for x in outputs])
        return model

    def test_output(self):
        cell_state_sizes = [[16, 16], [16, 16]]
        expected_output_shape = [(32, 16), (32, 16), (32, 16), (32, 16)]
        model = self.get_model(cell_state_sizes)
        output_shapes = [tensor.shape for tensor in model.output]
        self.assertTrue(expected_output_shape == output_shapes)
        input_data_sample = self.get_input_data_sample()
        output = model.predict(input_data_sample)
        output_shapes = [tensor.shape for tensor in output]
        self.assertTrue(expected_output_shape == output_shapes)

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

    def get_input_data_sample(self):
        return keras.ops.ones((self.batch_size, self.sequence_length, self.sequence_features))

    def get_model(self, layers_sizes, rnn_cell=None, merge_modes="concat", return_sequences=False, return_state=False):
        inputs = keras.layers.Input(shape=(self.sequence_length, self.sequence_features), batch_size=self.batch_size)
        outputs = rnn_layers.StackedBidirectionalRNN(
            layers_sizes=layers_sizes,
            rnn_cell=rnn_cell,
            merge_modes=merge_modes,
            return_sequences=return_sequences,
            return_state=return_state
        )(inputs)
        if return_state:
            outputs, *layer_states = outputs
            model = keras.models.Model(inputs=inputs, outputs=[outputs] + [state for state in layer_states])
        else:
            model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def test_output_predict(self):
        layers_sizes = [16, 32]
        input_data_sample = self.get_input_data_sample()
        expected_out_shape = self.batch_size, layers_sizes[-1]*2
        model = self.get_model(layers_sizes=layers_sizes, merge_modes="concat")
        self.assertTrue(expected_out_shape == model.output.shape)
        outputs = model.predict(input_data_sample)
        self.assertTrue(outputs.shape == expected_out_shape)

    def test_output_predict_with_none_merge_mode(self):
        layers_sizes = [16, 32]
        input_data_sample = self.get_input_data_sample()
        expected_out_shape = self.batch_size, layers_sizes[-1]*2
        model = self.get_model(layers_sizes=layers_sizes, merge_modes=None)
        self.assertTrue(len(model.output) == 2)
        self.assertTrue(expected_out_shape == s.shape for s in model.output)
        outputs = model.predict(input_data_sample)
        self.assertTrue(len(outputs) == 2)
        self.assertTrue(expected_out_shape == s.shape for s in outputs)

    def test_return_state_predict(self):
        layers_sizes = [16, 32]
        input_data_sample = self.get_input_data_sample()
        expected_out_shape = self.batch_size, layers_sizes[-1]*2
        expected_state_shape = (self.batch_size, layers_sizes[-1])
        model = self.get_model(layers_sizes=layers_sizes, return_state=True, merge_modes="concat")
        output, *states = model.output
        self.assertTrue(expected_out_shape == output.shape)
        self.assertTrue(len(states) == 4)
        self.assertTrue(expected_state_shape == x.shape for x in states)
        outputs, *states = model.predict(input_data_sample)
        self.assertTrue(expected_out_shape == output.shape)
        self.assertTrue(len(states) == 4)
        self.assertTrue(expected_state_shape == x.shape for x in states)

    def test_return_state_predict_with_none_merge_mode(self):
        layers_sizes = [16, 32]
        input_data_sample = self.get_input_data_sample()
        model = self.get_model(layers_sizes=layers_sizes, return_state=True, merge_modes=None)
        outputs, states = model.output[:2], model.output[2:]
        self.assertTrue(len(outputs) == 2)
        self.assertTrue((self.batch_size, layers_sizes[-1]) == x.shape for x in outputs)
        self.assertTrue(len(states) == 4)
        self.assertTrue((self.batch_size, layers_sizes[-1]) == x.shape for x in states)
        predictions = model.predict(input_data_sample)
        outputs, states = predictions[:2], predictions[2:]
        self.assertTrue(len(outputs) == 2)
        self.assertTrue((self.batch_size, layers_sizes[-1]) == x.shape for x in outputs)
        self.assertTrue(len(states) == 4)
        self.assertTrue((self.batch_size, layers_sizes[-1]) == x.shape for x in states)

    def test_return_sequences_predict(self):
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
