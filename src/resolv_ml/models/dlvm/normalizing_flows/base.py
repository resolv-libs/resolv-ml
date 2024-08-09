from typing import List

import keras
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from ....utilities.bijectors.base import Bijector
from ....utilities.schedulers import Scheduler


@keras.saving.register_keras_serializable(package="NormalizingFlows", name="NormalizingFlow")
class NormalizingFlow(keras.Model):

    def __init__(self,
                 bijectors: List[Bijector],
                 nll_weight_scheduler: Scheduler = None,
                 add_loss: bool = True,
                 name: str = "normalizing_flow",
                 **kwargs):
        super(NormalizingFlow, self).__init__(name=name, **kwargs)
        self._bijectors = bijectors
        self._bijectors_chain = tfb.Chain(bijectors)
        self._nll_weight_scheduler = nll_weight_scheduler
        self._add_loss = add_loss
        if self._add_loss:
            self._nll_loss_tracker = keras.metrics.Mean(name="nf_nll_loss")
            self._nll_weight_tracker = keras.metrics.Mean(name=f"nf_nll_weight")
            self._nll_weighted_loss_tracker = keras.metrics.Mean(name=f"nf_nll_loss_weighted")

    def build(self, input_shape):
        super().build(input_shape)
        for bijector in self._bijectors:
            if not bijector.built:
                bijector.build(input_shape)

    def call(self,
             inputs,
             base_distribution: tfd.Distribution = None,
             current_step=None,
             inverse: bool = False,
             training: bool = False,
             **kwargs):
        if self._add_loss:
            flow = tfd.TransformedDistribution(distribution=base_distribution, bijector=self._bijectors_chain)
            log_likelihood = flow.log_prob(inputs)
            negative_log_likelihood = -keras.ops.mean(log_likelihood)
            nll_weight = self._nll_weight_scheduler(step=current_step) if training \
                else self._nll_weight_scheduler.final_value()
            weighted_nll_loss = negative_log_likelihood * nll_weight
            self._nll_loss_tracker.update_state(negative_log_likelihood)
            self._nll_weight_tracker.update_state(nll_weight)
            self._nll_weighted_loss_tracker.update_state(weighted_nll_loss)
            self.add_loss(weighted_nll_loss)
        return self._bijectors_chain.inverse(inputs, **kwargs) if inverse \
            else self._bijectors_chain.forward(inputs, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "bijectors": keras.saving.serialize_keras_object(self._bijectors),
            "nll_weight_scheduler": keras.saving.serialize_keras_object(self._nll_weight_scheduler)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        bijectors = keras.saving.deserialize_keras_object(config.pop("bijectors"))
        nll_weight_scheduler = keras.saving.deserialize_keras_object(config.pop("nll_weight_scheduler"))
        return cls(bijectors=bijectors, nll_weight_scheduler=nll_weight_scheduler, **config)
