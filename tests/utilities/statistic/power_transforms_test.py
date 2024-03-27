import unittest
from pathlib import Path

import keras
import keras.ops as k_ops
import matplotlib as mpl
import matplotlib.pyplot as plt

from resolv_ml.utilities.statistic.power_transforms import BoxCox, YeoJohnson


class PowerTransformTest(unittest.TestCase):

    @property
    def box_cox_lambda(self) -> float:
        return 0.33

    @property
    def box_cox_layer(self) -> BoxCox:
        box_cox_layer = BoxCox(lambda_init=self.box_cox_lambda)
        box_cox_layer.trainable = False
        return box_cox_layer

    @property
    def yeo_johnson_lambda(self) -> YeoJohnson:
        return -2.22

    @property
    def yeo_johnson_layer(self) -> BoxCox:
        yeo_johnson_layer = YeoJohnson(lambda_init=self.yeo_johnson_lambda)
        yeo_johnson_layer.trainable = False
        return yeo_johnson_layer

    @property
    def input_distribution(self):
        return keras.random.beta(shape=(1000,), alpha=2, beta=6)

    @property
    def inverse_input_distribution(self):
        return keras.random.normal(shape=(1000,), mean=0, stddev=1)

    @property
    def output_dir(self) -> Path:
        return Path("./output")

    def setUp(self):
        self.output_dir.mkdir(exist_ok=True)

    def test_box_cox_transform(self):
        x = self.input_distribution
        y = self.box_cox_layer(x)
        y_std = k_ops.divide(k_ops.subtract(y, k_ops.mean(y)), k_ops.std(y))
        self._plot_distributions(x, y_std, self.box_cox_lambda, "box-cox-example")

    def test_box_cox_inverse_transform(self):
        x = self.inverse_input_distribution
        y = self.box_cox_layer(x, inverse=True)
        self._plot_distributions(x, y, self.box_cox_lambda, "box-cox-inverse-example")

    def test_yeo_johnson_transform(self):
        x = self.input_distribution
        y = self.yeo_johnson_layer(x)
        y_std = k_ops.divide(k_ops.subtract(y, k_ops.mean(y)), k_ops.std(y))
        self._plot_distributions(x, y_std, self.yeo_johnson_lambda, "yeo-johnson-example")

    def test_yeo_johnson_inverse_transform(self):
        x = self.inverse_input_distribution
        y = self.yeo_johnson_layer(x, inverse=True)
        self._plot_distributions(x, y, self.yeo_johnson_lambda, "yeo-johnson-inverse-example")

    @mpl.rc_context(rc={'text.usetex': True, 'font.family': 'serif', 'font.size': 18,
                        'font.serif': 'Computer Modern Roman', 'lines.linewidth': 2})
    def _plot_distributions(self, x, y, lmbda: float, output_fig_name: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)
        ax1.hist(x)
        ax2.hist(y)
        ax1.set_axisbelow(True)
        ax1.grid(linestyle=':')
        ax2.set_axisbelow(True)
        ax2.grid(linestyle=':')
        ax1.set_ylabel('Input distribution')
        ax2.set_ylabel(r'Output distribution ($\lambda$ = ' + f'{lmbda:.2f})')
        plt.savefig(self.output_dir / f'{output_fig_name}.png', dpi=300)
        plt.close()


if __name__ == '__main__':
    unittest.main()
