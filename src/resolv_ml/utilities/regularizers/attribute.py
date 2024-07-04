# TODO - DOC
from typing import List

import keras
from keras import ops as k_ops
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from .base import Regularizer
from ..bijectors.base import Bijector
from ..schedulers import Scheduler
from ...utilities.math.distances import compute_pairwise_distance_matrix


class AttributeRegularizer(Regularizer):

    def __init__(self,
                 weight_scheduler: Scheduler = None,
                 regularization_dimension: int = 0,
                 name: str = "attr_regularizer",
                 **kwargs):
        super(AttributeRegularizer, self).__init__(
            weight_scheduler=weight_scheduler,
            name=name,
            **kwargs
        )
        self._regularization_dimension = regularization_dimension

    def _compute_attribute_regularization_loss(self,
                                               latent_codes,
                                               attributes,
                                               prior: tfd.Distribution,
                                               posterior: tfd.Distribution,
                                               current_step=None,
                                               training: bool = False,
                                               evaluate: bool = False):
        raise NotImplementedError("_compute_attribute_regularization_loss must be implemented by subclasses.")

    def build(self, input_shape):
        super().build(input_shape)

    def _compute_regularization_loss(self,
                                     inputs,
                                     prior: tfd.Distribution,
                                     posterior: tfd.Distribution,
                                     current_step=None,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        _, attributes, z, _ = inputs
        latent_codes = k_ops.expand_dims(z[:, self._regularization_dimension], axis=-1)
        reg_loss = self._compute_attribute_regularization_loss(latent_codes, attributes, prior, posterior,
                                                               current_step, training, evaluate)
        return reg_loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "regularization_dimension": self._regularization_dimension
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="AttributeRegularizer", name="DefaultAttributeRegularizer")
class DefaultAttributeRegularizer(AttributeRegularizer):

    def __init__(self,
                 weight_scheduler: Scheduler = None,
                 loss_fn: keras.losses.Loss = keras.losses.MeanAbsoluteError(),
                 regularization_dimension: int = 0,
                 name: str = "default_attr_reg",
                 **kwargs):
        super(DefaultAttributeRegularizer, self).__init__(
            weight_scheduler=weight_scheduler,
            regularization_dimension=regularization_dimension,
            name=name,
            **kwargs
        )
        self._loss_fn = loss_fn

    def _compute_attribute_regularization_loss(self,
                                               latent_codes,
                                               attributes,
                                               prior: tfd.Distribution,
                                               posterior: tfd.Distribution,
                                               current_step=None,
                                               training: bool = False,
                                               evaluate: bool = False):
        return self._loss_fn(latent_codes, attributes)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(loss_fn=loss_fn, **config)


@keras.saving.register_keras_serializable(package="AttributeRegularizer", name="SignAttributeRegularizer")
class SignAttributeRegularizer(AttributeRegularizer):

    def __init__(self,
                 weight_scheduler: Scheduler = None,
                 loss_fn: keras.losses.Loss = keras.losses.MeanAbsoluteError(),
                 regularization_dimension: int = 0,
                 scale_factor: float = 1.0,
                 name: str = "sign_attr_reg",
                 **kwargs):
        super(SignAttributeRegularizer, self).__init__(
            weight_scheduler=weight_scheduler,
            regularization_dimension=regularization_dimension,
            name=name,
            **kwargs
        )
        self._loss_fn = loss_fn
        self._scale_factor = scale_factor

    def _compute_attribute_regularization_loss(self,
                                               latent_codes,
                                               attributes,
                                               prior: tfd.Distribution,
                                               posterior: tfd.Distribution,
                                               current_step=None,
                                               training: bool = False,
                                               evaluate: bool = False):
        lc_dist_mat = compute_pairwise_distance_matrix(latent_codes)
        attribute_dist_mat = compute_pairwise_distance_matrix(attributes)
        lc_tanh = k_ops.tanh(lc_dist_mat * self._scale_factor)
        attribute_sign = k_ops.sign(attribute_dist_mat)
        lc_tanh_reshaped = k_ops.reshape(lc_tanh, (-1,))
        attribute_sign_reshaped = k_ops.reshape(attribute_sign, (-1,))
        sign_loss = self._loss_fn(lc_tanh_reshaped, attribute_sign_reshaped)
        return sign_loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "scale_factor": self._scale_factor,
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(loss_fn=loss_fn, **config)


@keras.saving.register_keras_serializable(package="AttributeRegularizer", name="NormalizingFlow")
class NormalizingFlowAttributeRegularizer(AttributeRegularizer):

    def __init__(self,
                 bijectors: List[Bijector],
                 loss_fn: keras.losses.Loss = keras.losses.MeanAbsoluteError(),
                 regularization_dimension: int = 0,
                 reg_weight_scheduler: Scheduler = None,
                 nll_weight_scheduler: Scheduler = None,
                 name: str = "nf_attr_reg",
                 **kwargs):
        super(NormalizingFlowAttributeRegularizer, self).__init__(
            weight_scheduler=reg_weight_scheduler,
            regularization_dimension=regularization_dimension,
            name=name,
            **kwargs
        )
        self._bijectors = bijectors
        self._bijectors_chain = tfb.Chain(bijectors)
        self._loss_fn = loss_fn
        self._nll_weight_scheduler = nll_weight_scheduler
        self._nll_loss_tracker = keras.metrics.Mean(name="nf_nll_loss")
        self._nll_weight_tracker = keras.metrics.Mean(name=f"nf_nll_weight")
        self._nll_weighted_loss_tracker = keras.metrics.Mean(name=f"nf_nll_loss_weighted")

    def _compute_regularization_loss(self,
                                     inputs,
                                     prior: tfd.Distribution,
                                     posterior: tfd.Distribution,
                                     current_step=None,
                                     training: bool = False,
                                     evaluate: bool = False,
                                     **kwargs):
        _, attributes, z, _ = inputs
        # Build the flow that has the chosen regularization dimension component of the posterior distribution as base
        # distribution. For now, we assume that the posterior is a MultivariateGaussian.
        normalizing_flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
                loc=posterior.loc[:, self._regularization_dimension],
                scale_diag=posterior.scale.diag[:, self._regularization_dimension]),
            bijector=self._bijectors_chain
        )
        log_likelihood = normalizing_flow.log_prob(attributes)
        negative_log_likelihood = -k_ops.mean(log_likelihood)
        nll_weight = self._nll_weight_scheduler(step=current_step) if training \
            else self._nll_weight_scheduler.final_value()
        weighted_nll_loss = negative_log_likelihood * nll_weight
        self._nll_loss_tracker.update_state(negative_log_likelihood)
        self._nll_weight_tracker.update_state(nll_weight)
        self._nll_weighted_loss_tracker.update_state(weighted_nll_loss)
        self.add_loss(negative_log_likelihood)
        transformed_attributes = self._bijectors_chain.inverse(attributes)
        return self._loss_fn(z, transformed_attributes)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "bijectors": keras.saving.serialize_keras_object(self._bijectors),
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        bijectors = keras.saving.deserialize_keras_object(config.pop("bijectors"))
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(loss_fn=loss_fn, bijectors=bijectors, **config)
