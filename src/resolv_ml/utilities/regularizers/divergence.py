""" TODO - DOC """
from typing import Callable

import keras
from tensorflow_probability import distributions as tfd

from ..schedulers import Scheduler
from ...utilities.regularizers.base import Regularizer


@keras.saving.register_keras_serializable(package="Regularizers", name="Divergence")
class DivergenceRegularizer(Regularizer):

    def __init__(self,
                 divergence_fn: Callable[[tfd.Distribution, tfd.Distribution], float],
                 weight_scheduler: Scheduler = None,
                 free_bits: float = 0.0,
                 name: str = "div_regularizer",
                 **kwargs):
        super(DivergenceRegularizer, self).__init__(
            weight_scheduler=weight_scheduler,
            name=name,
            **kwargs
        )
        self._divergence_fn = divergence_fn
        self._free_bits = free_bits

    def _compute_regularization_loss(self,
                                     inputs,
                                     prior: tfd.Distribution,
                                     posterior: tfd.Distribution,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        div = self._divergence_fn(prior, posterior)
        # Compute the cost according to free_bits
        free_nats = self._free_bits * keras.ops.log(2.0)
        div_cost = keras.ops.maximum(div - free_nats, 0)
        # Compute KL across the batch
        div_loss = keras.ops.mean(div_cost)
        return div_loss
