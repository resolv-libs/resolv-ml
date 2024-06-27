import keras

from resolv_ml.utilities.bijectors.base import Bijector


@keras.saving.register_keras_serializable(package="Bijectors", name="PowerTransform")
class BoxCox(Bijector):

    def __init__(
            self,
            power: float = 0.0,
            shift: float = 0.0,
            power_trainable: bool = True,
            shift_trainable: bool = True,
            validate_args: bool = False,
            name: str = 'power_transform'):
        parameters = dict(locals())
        if power is None or power < 0:
            raise ValueError('`power` must be a non-negative constant.')
        self._power = keras.Variable(initializer=power, trainable=power_trainable, name="power")
        self._shift = keras.Variable(initializer=shift, trainable=shift_trainable, name="shift")
        super(BoxCox, self).__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            parameters=parameters,
            name=name)

    @property
    def power(self):
        return self._power

    @property
    def shift(self):
        return self._shift

    @classmethod
    def _parameter_properties(cls, dtype):
        return dict()

    @classmethod
    def _is_increasing(cls):
        return True

    def _forward(self, x):
        self._maybe_assert_valid_x(x)
        return keras.ops.cond(self.power == 0.,
                              true_fn=lambda: keras.ops.exp(x) - self.shift,
                              false_fn=lambda: (1. + x * self.power)**(1. / self.power) - self.shift)

    def _inverse(self, y):
        self._maybe_assert_valid_y(y)
        shifted_y = y + self.shift
        return keras.ops.cond(self.power == 0.,
                              true_fn=lambda: keras.ops.log(shifted_y),
                              false_fn=lambda: (shifted_y**self.power - 1.) / self.power)

    def _inverse_log_det_jacobian(self, y):
        self._maybe_assert_valid_y(y)
        shifted_y = y + self.shift
        return keras.ops.cond(self.power == 0.,
                              true_fn=lambda: -keras.ops.log(shifted_y),
                              false_fn=lambda: (self.power - 1) * keras.ops.log(shifted_y))

    def _forward_log_det_jacobian(self, x):
        self._maybe_assert_valid_x(x)
        return keras.ops.cond(self.power == 0.,
                              true_fn=lambda: x,
                              false_fn=lambda: (keras.ops.reciprocal(self.power) - 1) * keras.ops.log1p(x * self.power))

    def _maybe_assert_valid_x(self, x):
        if self.validate_args and not self.power == 0.:
            assert 1. + self.power * x - self.shift ** self.power > 0, \
                f'Forward transformation input must be at least {(self.shift ** self.power - 1.) / self.power}.'

    def _maybe_assert_valid_y(self, y):
        if self.validate_args:
            assert y + self.shift > 0, f'Inverse transformation input must be greater than or equal to {self.shift}.'

    def get_config(self):
        base_config = super().get_config()
        config = {
            "power": keras.saving.serialize_keras_object(self.power),
            "shift": keras.saving.serialize_keras_object(self.shift)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        power = keras.saving.deserialize_keras_object(config.pop("power"))
        shift = keras.saving.deserialize_keras_object(config.pop("shift"))
        return cls(power, shift, **config)
