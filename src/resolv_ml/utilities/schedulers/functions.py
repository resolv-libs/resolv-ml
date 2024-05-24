import keras

from .base import Scheduler, FunctionScheduler


@keras.saving.register_keras_serializable(package="Schedulers", name="ConstantScheduler")
class ConstantScheduler(Scheduler):

    def __init__(self, value: float, name="constant_scheduler"):
        super(ConstantScheduler, self).__init__(name=name)
        self._value = value

    def __call__(self, step: int):
        return self._value

    def final_value(self) -> float:
        return self._value

    def get_config(self):
        base_config = super().get_config()
        return {
            "value": self._value,
            **base_config
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="Schedulers", name="LinearScheduler")
class LinearScheduler(FunctionScheduler):

    def __init__(self,
                 rate: float = None,
                 min_value: float = 0.0,
                 max_value: float = None,
                 decay: bool = False,
                 total_steps: int = None,
                 name: str = "linear_scheduler"):
        super(LinearScheduler, self).__init__(
            rate=rate,
            min_value=min_value,
            max_value=max_value,
            decay=decay,
            total_steps=total_steps,
            name=name
        )

    def _growth_fn(self, step: int) -> float:
        return min(self._min_value + step * self._rate, self._max_value)

    def _decay_fn(self, step: int) -> float:
        return max(self._max_value - step * self._rate, self._min_value)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="Schedulers", name="ExponentialScheduler")
class ExponentialScheduler(FunctionScheduler):

    def __init__(self,
                 rate: float = None,
                 min_value: float = 0.0,
                 max_value: float = None,
                 decay: bool = False,
                 name: str = "exponential_scheduler"):
        if not 0 <= rate < 1:
            raise ValueError(f'`exponential_scheduler` rate must be in the interval [0, 1). Got {rate}.')
        super(ExponentialScheduler, self).__init__(
            rate=rate,
            min_value=min_value,
            max_value=max_value,
            decay=decay,
            name=name
        )

    def _growth_fn(self, step: int) -> float:
        return (1 - keras.ops.power(self._rate, step)) * (self._max_value - self._min_value) + self._min_value

    def _decay_fn(self, step: int) -> float:
        return keras.ops.power(self._rate, step) * (self._max_value - self._min_value) + self._min_value

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="Schedulers", name="SigmoidScheduler")
class SigmoidScheduler(FunctionScheduler):

    def __init__(self,
                 rate: float = None,
                 min_value: float = 0.0,
                 max_value: float = None,
                 decay: bool = False,
                 name: str = "sigmoid_scheduler"):
        if rate < 1:
            raise ValueError(f'`sigmoid_scheduler` rate must be at least 1. Got {rate}.')
        super(SigmoidScheduler, self).__init__(
            rate=rate,
            min_value=min_value,
            max_value=max_value,
            decay=decay,
            name=name
        )

    def _growth_fn(self, step: int) -> float:
        return 1 - self._decay_fn(step)

    def _decay_fn(self, step: int) -> float:
        return ((self._rate + 1) / (self._rate + keras.ops.exp(step / self._rate)) *
                (self._max_value - self._min_value) + self._min_value)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
