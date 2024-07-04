import keras

from resolv_ml.utilities.bijectors.base import Bijector


@keras.saving.register_keras_serializable(package="Bijectors", name="PowerTransform")
class BoxCox(Bijector):

    def __init__(
            self,
            power_init_value: float = 0.0,
            shift_init_value: float = 0.0,
            power_trainable: bool = True,
            shift_trainable: bool = True,
            validate_args: bool = False,
            name: str = 'power_transform'):
        super(BoxCox, self).__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            parameters=dict(locals()),
            name=name)
        self._power_init_value = power_init_value
        self._shift_init_value = shift_init_value
        self._power_trainable = power_trainable
        self._shift_trainable = shift_trainable

    def build(self, input_shape):
        super().build(input_shape)
        self._power = self.add_weight(
            initializer=keras.initializers.Constant(self._power_init_value),
            trainable=self._power_trainable,
            name='power'
        )
        self._shift = self.add_weight(
            initializer=keras.initializers.Constant(self._shift_init_value),
            trainable=self._shift_trainable,
            name='shift'
        )

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
