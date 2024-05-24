# TODO - Cyclical scheduler implementation (see https://arxiv.org/abs/1903.10145)
import keras


@keras.saving.register_keras_serializable(package="Schedulers", name="Scheduler")
class Scheduler:

    def __init__(self, name: str = "scheduler"):
        self._name = name

    def __call__(self, step):
        raise NotImplementedError(f"Scheduler class '{self.__class__.__name__}' must override `__call__(self, step)`.")

    def final_value(self) -> float:
        raise NotImplementedError(f"Scheduler class '{self.__class__.__name__}' must override `final_value`.")

    def get_config(self):
        return {
            "name": self._name
        }


@keras.saving.register_keras_serializable(package="Schedulers", name="FunctionScheduler")
class FunctionScheduler(Scheduler):

    def __init__(self,
                 rate: float,
                 min_value: float = 0.0,
                 max_value: float = None,
                 decay: bool = False,
                 total_steps: int = None,
                 name: str = "function_scheduler"):
        super().__init__(name=name)
        if not max_value and decay:
            raise ValueError("`max_value` must be provided if `decay` is true.")
        self._rate = rate
        self._min_value = min_value
        self._max_value = max_value
        self._decay = decay
        self._total_steps = total_steps

    def __call__(self, step: int) -> float:
        return self._decay_fn(step) if self._decay else self._growth_fn(step)

    def final_value(self) -> float:
        if self._total_steps:
            return self._decay_fn(self._total_steps) if self._decay else self._growth_fn(self._total_steps)
        else:
            return self._min_value if self._decay else self._max_value

    def _growth_fn(self, step: int) -> float:
        raise NotImplementedError(f"Scheduler class '{self.__class__.__name__}' must override `_growth_fn(self)`.")

    def _decay_fn(self, step: int) -> float:
        raise NotImplementedError(f"Scheduler class '{self.__class__.__name__}' must override `_decay_fn(self)`.")

    def get_config(self):
        base_config = super().get_config()
        return {
            "min_value": self._min_value,
            "max_value": self._max_value,
            "rate": self._rate,
            "decay": self._decay,
            "total_steps": self._total_steps,
            **base_config
        }
