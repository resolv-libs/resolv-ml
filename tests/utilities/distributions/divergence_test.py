import os
import unittest
from pathlib import Path

import keras
import numpy
from deepdiff import DeepDiff

from resolv_ml.utilities.distributions import divergence as div_utils


class TestBetaDivergenceRegularizer(unittest.TestCase):

    @property
    def output_dir(self):
        return Path("./output/utilities/distributions/divergence")

    @property
    def batch_size(self):
        return 32

    @property
    def input_size(self):
        return 70

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_input_data_sample(self):
        empty = keras.ops.zeros((1,))
        p_mean = keras.random.uniform((self.batch_size, self.input_size))
        p_var = keras.random.uniform((self.batch_size, self.input_size))
        return empty, empty, (p_mean, p_var), empty, empty

    def get_model(self, max_beta=1.0, beta_rate=0.0, free_bits=0.0):
        not_used = keras.layers.Input(shape=(1,), batch_size=self.batch_size)
        p_mean_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        p_var_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        outputs = div_utils.BetaDivergenceRegularizer(
            divergence_layer=div_utils.GaussianKLDivergence(),
            max_beta=max_beta,
            beta_rate=beta_rate,
            free_bits=free_bits
        )((not_used, not_used, (p_mean_input, p_var_input), not_used, not_used))
        return keras.Model(inputs=[not_used, not_used, p_mean_input, p_var_input, not_used, not_used],
                           outputs=outputs)

    def get_layer(self, max_beta=1.0, beta_rate=0.0, free_bits=0.0):
        return div_utils.BetaDivergenceRegularizer(
            divergence_layer=div_utils.GaussianKLDivergence(),
            max_beta=max_beta,
            beta_rate=beta_rate,
            free_bits=free_bits
        )

    def test_beta_rate_check(self):
        div_layer = div_utils.GaussianKLDivergence()
        self.assertRaises(ValueError, lambda: div_utils.BetaDivergenceRegularizer(div_layer, beta_rate=-2.0))
        self.assertRaises(ValueError, lambda: div_utils.BetaDivergenceRegularizer(div_layer, beta_rate=+2.0))

    def test_predict_beta_rate_0(self):
        n_iterations = 50
        model = self.get_layer()
        outputs = []
        for i in range(n_iterations):
            input_data_sample = self.get_input_data_sample()
            output = model(input_data_sample, iterations=i+1)
            outputs.append(output)
        self.assertTrue(outputs)

    def test_predict_beta_rate(self):
        n_iterations = 50
        model = self.get_layer(beta_rate=0.5)
        outputs = []
        for i in range(n_iterations):
            input_data_sample = self.get_input_data_sample()
            output = model(input_data_sample, iterations=i+1)
            outputs.append(output)
        self.assertTrue(outputs)

    def test_predict_free_bits(self):
        n_iterations = 50
        model = self.get_layer(free_bits=0.9999)
        outputs = []
        for i in range(n_iterations):
            input_data_sample = self.get_input_data_sample()
            output = model(input_data_sample, iterations=i+1)
            outputs.append(output)
        self.assertTrue(outputs)

    def test_saving_and_loading(self):
        model = self.get_model()
        model.save(self.output_dir / "beta_divergence_reg.keras")
        loaded_model = keras.saving.load_model(self.output_dir / "beta_divergence_reg.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


class TestGaussianKLDivergence(unittest.TestCase):

    @property
    def output_dir(self):
        return Path("./output/utilities/distributions/divergence")

    @property
    def batch_size(self):
        return 32

    @property
    def input_size(self):
        return 70

    def setUp(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_input_data_sample(self, empty_q=True):
        p_mean = keras.random.uniform((self.batch_size, self.input_size))
        p_var = keras.random.uniform((self.batch_size, self.input_size))
        if empty_q:
            return p_mean, p_var
        q_mean = keras.random.uniform((self.batch_size, self.input_size))
        q_var = keras.random.uniform((self.batch_size, self.input_size))
        return p_mean, p_var, q_mean, q_var

    def get_model(self):
        p_mean_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        p_var_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        q_mean_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        q_var_input = keras.Input(shape=(self.input_size,), batch_size=self.batch_size)
        output = div_utils.GaussianKLDivergence()((p_mean_input, p_var_input, q_mean_input, q_var_input))
        return keras.Model(inputs=[p_mean_input, p_var_input, q_mean_input, q_var_input], outputs=output)

    def test_predict(self):
        model = div_utils.GaussianKLDivergence()
        input_data_sample = self.get_input_data_sample(empty_q=False)
        outputs = model(input_data_sample)
        self.assertTrue(isinstance(outputs.numpy(), numpy.float32))

    def test_predict_default_input(self):
        model = div_utils.GaussianKLDivergence()
        input_data_sample = self.get_input_data_sample()
        outputs = model(input_data_sample)
        self.assertTrue(isinstance(outputs.numpy(), numpy.float32))

    def test_saving_and_loading(self):
        model = self.get_model()
        model.save(self.output_dir / "gaussian_kl_divergence.keras")
        loaded_model = keras.saving.load_model(self.output_dir / "gaussian_kl_divergence.keras")
        # Use DeepDiff to ignore tuple to list type change in config comparison
        diff = DeepDiff(loaded_model.get_config(), model.get_config(), ignore_type_in_groups=(list, tuple))
        self.assertTrue(not diff)


if __name__ == '__main__':
    unittest.main()
