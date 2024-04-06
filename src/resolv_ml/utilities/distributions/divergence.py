from typing import Tuple, Any

import keras
import keras.ops as k_ops


class BetaDivergence(keras.Model):

    def __init__(self,
                 divergence_layer: keras.Layer,
                 max_beta: float = None,
                 beta_rate: float = None,
                 free_bits: float = None,
                 **kwargs):
        super(BetaDivergence, self).__init__(name=divergence_layer.name, **kwargs)
        self._divergence_layer = divergence_layer
        self._free_bits = free_bits
        self._max_beta = max_beta
        self._beta_rate = beta_rate

    @property
    def metrics(self):
        return [
            self.div_loss_tracker,
            self.div_bits_tracker,
            self.div_beta_tracker
        ]

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        self.div_loss_tracker = keras.metrics.Mean(name=f"{self._divergence_layer.name}_loss")
        self.div_bits_tracker = keras.metrics.Mean(name=f"{self._divergence_layer.name}_bits")
        self.div_beta_tracker = keras.metrics.Mean(name=f"{self._divergence_layer.name}_beta")

    def call(self, inputs: Any, training: bool = False, **kwargs):
        div = self._divergence_layer(inputs)
        # Compute the cost according to free_bits
        free_nats = self._free_bits * k_ops.log(2.0)
        div_cost = k_ops.maximum(div - free_nats, 0)
        # Compute beta for beta-VAE
        div_beta = (1.0 - k_ops.power(self._beta_rate, self.optimizer.iterations)) * self._max_beta
        # Compute KL across the batch and update trackers
        div_loss = div_beta * k_ops.mean(div_cost)
        div_loss_bits = div_loss / k_ops.log(2.0)
        self.add_loss(div_loss)
        self.div_loss_tracker.update_state(div_loss)
        self.div_bits_tracker.update_state(div_loss_bits)
        self.div_beta_tracker.update_state(div_beta)


class GaussianKLDivergence(keras.Layer):

    def __init__(self, name: str = "gauss_kl_div", **kwargs):
        super(GaussianKLDivergence, self).__init__(name=name, **kwargs)

    def call(self, p_mean, p_var, q_mean=0, q_var=1, training: bool = False, **kwargs):
        kl_loss = 0.5 * (k_ops.log(p_var/q_var) + (k_ops.square(q_mean-p_mean) + q_var)/p_var - 1)
        return k_ops.sum(kl_loss, axis=-1)
