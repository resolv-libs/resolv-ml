import keras

from .base import Bijector


@keras.saving.register_keras_serializable(package="Bijectors", name="BatchNormalization")
class BatchNormalization(Bijector):

    def __init__(self,
                 batchnorm_layer: keras.layers.BatchNormalization = None,
                 training: bool = True,
                 center: bool = True,
                 scale: bool = True,
                 validate_args: bool = False,
                 name: str = 'batch_normalization'):
        parameters = dict(locals())
        self.batchnorm = batchnorm_layer or keras.layers.BatchNormalization(
            center=center,
            scale=scale
        )
        if not isinstance(self.batchnorm, keras.layers.BatchNormalization):
            raise ValueError(f'batchnorm_layer must be an instance of `keras.layers.BatchNormalization`. '
                             f'Got {type(self.batchnorm)}')
        self._training = training
        forward_min_event_ndims = 1 if isinstance(self.batchnorm.axis, int) else len(self.batchnorm.axis)
        super(BatchNormalization, self).__init__(
            forward_min_event_ndims=forward_min_event_ndims,
            validate_args=validate_args,
            parameters=parameters,
            name=name)

    @classmethod
    def _parameter_properties(cls, dtype):
        return dict()

    def _normalize(self, y):
        return self.batchnorm(y, training=self._training)

    def _de_normalize(self, x):
        def _undo_batch_normalization():
            rescale = keras.ops.sqrt(variance + self.batchnorm.epsilon)
            if gamma is not None:
                rescale = rescale / gamma
            batch_denormalized = x * rescale + (mean - beta * rescale if beta is not None else mean)
            return batch_denormalized

        # Uses the saved statistics.
        if not self.batchnorm.built:
            self.batchnorm.build(x.shape)
        mean = self.batchnorm.moving_mean
        variance = self.batchnorm.moving_variance
        beta = self.batchnorm.beta if self.batchnorm.center else None
        gamma = self.batchnorm.gamma if self.batchnorm.scale else None
        return _undo_batch_normalization()

    def _forward(self, x):
        return self._de_normalize(x)

    def _inverse(self, y, **kwargs):
        return self._normalize(y)

    def _forward_log_det_jacobian(self, x):
        # Uses saved statistics to compute volume distortion.
        return -self._inverse_log_det_jacobian(x, use_saved_statistics=True)

    def _inverse_log_det_jacobian(self, y, use_saved_statistics=False, **kwargs):
        if not self.batchnorm.built:
            self.batchnorm.build(y.shape)

        # At training-time, ildj is computed from the mean and log-variance across the current minibatch.
        log_variance = keras.ops.log(
            keras.ops.where(
                keras.ops.logical_or(use_saved_statistics, keras.ops.logical_not(self._training)),
                self.batchnorm.moving_variance,
                keras.ops.moments(x=y, axes=self.batchnorm.axis, keepdims=True)[1]
            ) + self.batchnorm.epsilon
        )

        # Log(total change in area from gamma term).
        log_total_gamma = keras.ops.sum(keras.ops.log(self.batchnorm.gamma)) if self.batchnorm.scale else 0.

        # Log(total change in area from log-variance term).
        log_total_variance = keras.ops.sum(log_variance)
        # The ildj is scalar, as it does not depend on the values of x and are constant across minibatch elements.
        return log_total_gamma - 0.5 * log_total_variance

    def get_config(self):
        base_config = super().get_config()
        config = {
            "batchnorm": keras.saving.serialize_keras_object(self.batchnorm),
            "training": self._training
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        batchnorm = keras.saving.deserialize_keras_object(config.pop("batchnorm"))
        return cls(batchnorm_layer=batchnorm, **config)
