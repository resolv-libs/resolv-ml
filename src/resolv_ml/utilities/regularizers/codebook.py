""" TODO - DOC """
import keras
from tensorflow_probability import distributions as tfd

from ..schedulers import Scheduler
from ...utilities.regularizers.base import Regularizer


@keras.saving.register_keras_serializable(package="Regularizers", name="VectorQuantization")
class VectorQuantizationRegularizer(Regularizer):

    def __init__(self,
                 weight_scheduler: Scheduler = None,
                 name: str = "vq_regularizer",
                 **kwargs):
        super(VectorQuantizationRegularizer, self).__init__(
            weight_scheduler=weight_scheduler,
            name=name,
            **kwargs
        )

    def _compute_regularization_loss(self,
                                     inputs,
                                     prior: tfd.Distribution,
                                     posterior: tfd.Distribution,
                                     current_step=None,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        _, _, z, input_features, _ = inputs
        vq_loss = keras.ops.mean(keras.ops.power(z - keras.ops.stop_gradient(input_features), 2))
        return vq_loss


@keras.saving.register_keras_serializable(package="Regularizers", name="Commitment")
class CommitmentRegularizer(Regularizer):

    def __init__(self,
                 weight_scheduler: Scheduler = None,
                 name: str = "commitment_regularizer",
                 **kwargs):
        super(CommitmentRegularizer, self).__init__(
            weight_scheduler=weight_scheduler,
            name=name,
            **kwargs
        )

    def _compute_regularization_loss(self,
                                     inputs,
                                     prior: tfd.Distribution,
                                     posterior: tfd.Distribution,
                                     current_step=None,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        _, _, z, input_features, _ = inputs
        commitment_loss = keras.ops.mean(keras.ops.power(keras.ops.stop_gradient(z) - input_features, 2))
        return commitment_loss
