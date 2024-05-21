# TODO - DOC
# TODO - add multi-backend support for probability distributions
import keras
from tensorflow_probability import distributions as tfp_dist

from resolv_ml.utilities.schedulers import Scheduler


class Regularizer(keras.Layer):

    def __init__(self,
                 beta_scheduler: Scheduler = None,
                 name: str = "regularizer",
                 **kwargs):
        super(Regularizer, self).__init__(name=name, **kwargs)
        self._beta_scheduler = beta_scheduler

    def call(self,
             inputs,
             posterior: tfp_dist.Distribution,
             iterations=None,
             training: bool = False,
             evaluate: bool = False,
             **kwargs):
        if training or evaluate:
            beta = self._beta_scheduler(step=iterations) if self._beta_scheduler and training else 1.0
            reg_loss = self._compute_regularization_loss(inputs, posterior, training, evaluate)
            return beta * reg_loss, beta
        else:
            raise RuntimeError("Regularizer layers can be called only in training or evaluate mode.")

    def compute_output_shape(self, input_shape):
        return (2,)

    def _compute_regularization_loss(self,
                                     inputs,
                                     posterior: tfp_dist.Distribution,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        raise NotImplementedError("_compute_regularization_loss must be implemented by subclasses.")
