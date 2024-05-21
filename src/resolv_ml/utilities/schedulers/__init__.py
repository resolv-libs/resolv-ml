from typing import Dict, Any

from .base import Scheduler
from .functions import ConstantScheduler, LinearScheduler, ExponentialScheduler, SigmoidScheduler

_SUPPORTED_SCHEDULES = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
    "exponential": ExponentialScheduler,
    "sigmoid": SigmoidScheduler
}


def get_scheduler(schedule_type: str, schedule_config: Dict[str, Any]) -> Scheduler:
    if schedule_type not in _SUPPORTED_SCHEDULES.keys():
        raise ValueError(f"Schedule type {schedule_type} not supported. Supported types are {_SUPPORTED_SCHEDULES}")
    return _SUPPORTED_SCHEDULES[schedule_type](**schedule_config)
