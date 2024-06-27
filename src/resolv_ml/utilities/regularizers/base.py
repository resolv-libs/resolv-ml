# TODO - DOC
# TODO - add multi-backend support for probability distributions
import keras
from tensorflow_probability import distributions as tfd

from resolv_ml.utilities.schedulers import Scheduler


class Regularizer(keras.Layer):

    def __init__(self,
                 weight_scheduler: Scheduler = None,
                 name: str = "regularizer",
                 **kwargs):
        super(Regularizer, self).__init__(name=name, **kwargs)
        self._weight_scheduler = weight_scheduler
        self._loss_tracker = keras.metrics.Mean(name=f"{name}_loss")
        self._weight_tracker = keras.metrics.Mean(name=f"{name}_weight")
        self._weighted_loss_tracker = keras.metrics.Mean(name=f"{name}_loss_weighted")

    def call(self,
             inputs,
             prior: tfd.Distribution,
             posterior: tfd.Distribution,
             current_step=None,
             training: bool = False,
             evaluate: bool = False,
             **kwargs):
        if training or evaluate:
            weight = 1.0
            if self._weight_scheduler:
                weight = self._weight_scheduler(step=current_step) if training else self._weight_scheduler.final_value()
            reg_loss = self._compute_regularization_loss(inputs, prior, posterior, training, evaluate)
            weighted_reg_loss = reg_loss * weight
            self._loss_tracker.update_state(reg_loss)
            self._weight_tracker.update_state(weight)
            self._weighted_loss_tracker.update_state(weighted_reg_loss)
            return weighted_reg_loss
        else:
            raise RuntimeError("Regularizer layers can be called only in training or evaluate mode.")

    def compute_output_shape(self, input_shape):
        return (1,)

    def _compute_regularization_loss(self,
                                     inputs,
                                     prior: tfd.Distribution,
                                     posterior: tfd.Distribution,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        raise NotImplementedError("_compute_regularization_loss must be implemented by subclasses.")
