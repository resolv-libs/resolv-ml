""" TODO - DOC """
import keras
from keras import ops as k_ops
from tensorflow_probability import distributions as tfp_dist

from ..schedulers import Scheduler
from ...utilities.regularizers.base import Regularizer


@keras.saving.register_keras_serializable(package="Regularizers", name="Divergence")
class DivergenceRegularizer(Regularizer):

    def __init__(self,
                 divergence_layer: keras.Layer,
                 beta_scheduler: Scheduler = None,
                 free_bits: float = 0.0,
                 name: str = "div_regularizer",
                 **kwargs):
        super(DivergenceRegularizer, self).__init__(
            beta_scheduler=beta_scheduler,
            name=name,
            **kwargs
        )
        self._divergence_layer = divergence_layer
        self._free_bits = free_bits

    def build(self, input_shape):
        super().build(input_shape)
        self._divergence_layer.build(input_shape)

    def _compute_regularization_loss(self,
                                     inputs,
                                     posterior: tfp_dist.Distribution,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        div = self._divergence_layer(inputs, posterior=posterior, training=training)
        # Compute the cost according to free_bits
        free_nats = self._free_bits * k_ops.log(2.0)
        div_cost = k_ops.maximum(div - free_nats, 0)
        # Compute KL across the batch
        div_loss = k_ops.mean(div_cost)
        return div_loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "divergence_layer": keras.saving.serialize_keras_object(self._divergence_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        divergence_layer = keras.saving.deserialize_keras_object(config.pop("divergence_layer"))
        return cls(divergence_layer=divergence_layer, **config)
