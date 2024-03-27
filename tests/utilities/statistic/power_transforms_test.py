import unittest
from pathlib import Path

import keras
import keras.ops as k_ops
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

from resolv_ml.utilities.statistic.power_transforms import BoxCox, YeoJohnson


class PowerTransformTest(unittest.TestCase):

    @property
    def input_distribution(self):
        return keras.random.beta(shape=(1000, 1), alpha=2, beta=6)

    @property
    def inverse_input_distribution(self):
        return keras.random.normal(shape=(1000, 1), mean=0, stddev=1)

    @property
    def output_dir(self) -> Path:
        return Path("./output/power-transforms")

    def setUp(self):
        self.output_dir.mkdir(exist_ok=True)

    @mpl.rc_context(rc={'text.usetex': True, 'font.family': 'serif', 'font.size': 18,
                        'font.serif': 'Computer Modern Roman', 'lines.linewidth': 2})
    def _plot_distributions(self, x, y, lmbda: float, output_fig_name: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)
        ax1.hist(k_ops.reshape(x, [-1]))
        ax2.hist(k_ops.reshape(y, [-1]))
        ax1.set_axisbelow(True)
        ax1.grid(linestyle=':')
        ax2.set_axisbelow(True)
        ax2.grid(linestyle=':')
        ax1.set_ylabel('Input distribution')
        ax2.set_ylabel(r'Output distribution ($\lambda$ = ' + f'{lmbda:.2f})')
        plt.savefig(self.output_dir / f'{output_fig_name}.png', dpi=300)
        plt.close()


class BoxCoxPowerTransformsTest(PowerTransformTest):

    @staticmethod
    def box_cox_layer(lmbda: float = 0.33) -> BoxCox:
        box_cox_layer = BoxCox(lambda_init=lmbda, batch_norm=keras.layers.BatchNormalization())
        box_cox_layer.trainable = False
        return box_cox_layer

    def test_box_cox(self):
        x = self.input_distribution
        y = self.box_cox_layer()(x)
        self._plot_distributions(x, y, 0.33, "box-cox")

    def test_box_cox_graph(self):
        @tf.function
        def graph_wrapper():
            return box_cox_layer(x)

        x = self.input_distribution
        box_cox_layer = self.box_cox_layer()
        y = graph_wrapper()
        self._plot_distributions(x, y, 0.33, "box-cox-graph")

    def test_box_cox_inverse(self):
        x = self.inverse_input_distribution
        y = self.box_cox_layer()(x, inverse=True)
        self._plot_distributions(x, y, 0.33, "box-cox-inverse")

    def test_box_cox_inverse_graph(self):
        @tf.function
        def graph_wrapper():
            return box_cox_layer(x, inverse=True)

        x = self.input_distribution
        box_cox_layer = self.box_cox_layer()
        y = graph_wrapper()
        self._plot_distributions(x, y, 0.33, "box-cox-inverse-graph")

    def test_box_cox_zero_lambda(self):
        x = self.input_distribution
        y = self.box_cox_layer(lmbda=0)(x)
        self._plot_distributions(x, y, 0, "box-cox-zero-lambda")

    def test_box_cox_inverse_zero_lambda(self):
        x = self.inverse_input_distribution
        y = self.box_cox_layer()(x, inverse=True)
        self._plot_distributions(x, y, 0, "box-cox-inverse-zero-lambda")

    def test_box_cox_transform_null_values(self):
        x = k_ops.convert_to_tensor([0, 10, 20])
        with self.assertRaises(ValueError):
            self.box_cox_layer()(x)

    def test_box_cox_transform_negative_values(self):
        x = k_ops.convert_to_tensor([-1, 10, 20])
        with self.assertRaises(ValueError):
            self.box_cox_layer()(x)


class YeoJohnsonPowerTransformsTest(PowerTransformTest):

    @staticmethod
    def yeo_johnson_layer(lmbda: float = -2.22) -> YeoJohnson:
        yeo_johnson_layer = YeoJohnson(lambda_init=lmbda, batch_norm=keras.layers.BatchNormalization())
        yeo_johnson_layer.trainable = False
        return yeo_johnson_layer

    def test_yeo_johnson(self):
        x = self.input_distribution
        y = self.yeo_johnson_layer()(x)
        self._plot_distributions(x, y, -2.22, "yeo-johnson")

    def test_yeo_johnson_inverse(self):
        x = self.inverse_input_distribution
        y = self.yeo_johnson_layer()(x, inverse=True)
        self._plot_distributions(x, y, -2.22, "yeo-johnson-inverse")

    def test_yeo_johnson_zero_lambda(self):
        x = self.input_distribution
        y = self.yeo_johnson_layer(lmbda=0)(x)
        self._plot_distributions(x, y, 0, "yeo-johnson-zero-lambda")

    def test_yeo_johnson_inverse_zero_lambda(self):
        x = self.inverse_input_distribution
        y = self.yeo_johnson_layer(lmbda=0)(x, inverse=True)
        self._plot_distributions(x, y, 0, "yeo-johnson-inverse-zero-lambda")

    def test_yeo_johnson_two_lambda(self):
        x = self.input_distribution
        y = self.yeo_johnson_layer(lmbda=2)(x)
        self._plot_distributions(x, y, 2, "yeo-johnson-two-lambda")

    def test_yeo_johnson_inverse_two_lambda(self):
        x = self.inverse_input_distribution
        y = self.yeo_johnson_layer(lmbda=2)(x, inverse=True)
        self._plot_distributions(x, y, 2, "yeo-johnson-inverse-two-lambda")


if __name__ == '__main__':
    unittest.main()
