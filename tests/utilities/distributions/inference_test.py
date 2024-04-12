import os
import unittest
from pathlib import Path

import keras
from deepdiff import DeepDiff

from resolv_ml.utilities.distributions import inference as inf_utils


class TestGaussianInference(unittest.TestCase):

    @property
    def output_dir(self):
        return Path("./output/utilities/distributions/divergence")

    @property
    def batch_size(self):
        return 32

    @property
    def input_size(self):
        return 70

    @property
    def z_size(self):
        return 128

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_input_data_sample(self):
        return keras.random.uniform((self.batch_size, self.input_size))

    def get_model(self):
        inputs = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        mean, log_var = inf_utils.GaussianInference(z_size=self.z_size)(inputs)
        return keras.Model(inputs=inputs, outputs=[mean, log_var])

    def test_predict(self):
        expected_output_shape = self.batch_size, self.input_size
        model = self.get_model()
        self.assertTrue(expected_output_shape == x.shape for x in model.output)
        input_data_sample = self.get_input_data_sample()
        outputs = model.predict(input_data_sample)
        self.assertTrue(expected_output_shape == x.shape for x in outputs)

    def test_saving_and_loading(self):
        model = self.get_model()
        model.save(self.output_dir / "beta_divergence_reg.keras")
        loaded_model = keras.saving.load_model(self.output_dir / "beta_divergence_reg.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


class TestGaussianReparametrizationTrick(unittest.TestCase):

    @property
    def output_dir(self):
        return Path("./output/utilities/distributions/inference")

    @property
    def batch_size(self):
        return 32

    @property
    def input_size(self):
        return 70

    @property
    def z_size(self):
        return 128

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_input_data_sample(self):
        p_mean = keras.random.uniform((self.batch_size, self.input_size))
        p_var = keras.random.uniform((self.batch_size, self.input_size))
        aux_input = keras.ops.zeros((self.batch_size, 1))
        return p_mean, p_var, aux_input

    def get_model(self):
        p_mean_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        p_var_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        aux_input = keras.Input(shape=(1,), batch_size=self.batch_size)
        output = inf_utils.GaussianReparametrizationTrick(z_size=self.z_size)((p_mean_input, p_var_input, aux_input))
        return keras.Model(inputs=[p_mean_input, p_var_input, aux_input], outputs=output)

    def test_predict(self):
        expected_output_shape = self.batch_size, self.input_size
        model = self.get_model()
        self.assertTrue(expected_output_shape == model.output.shape)
        input_data_sample = self.get_input_data_sample()
        outputs = model.predict(input_data_sample)
        self.assertTrue(outputs.shape == expected_output_shape)

    def test_saving_and_loading(self):
        model = self.get_model()
        model.save(self.output_dir / "gaussian_reparametrization_trick.keras")
        loaded_model = keras.saving.load_model(self.output_dir / "gaussian_reparametrization_trick.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


if __name__ == '__main__':
    unittest.main()
