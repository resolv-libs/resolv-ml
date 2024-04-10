import keras


def get_sampling_probability(step: int,
                             sampling_schedule: str = 'constant',
                             sampling_rate: float = 0.0,
                             training: bool = False) -> float:
    if sampling_schedule == 'constant' and sampling_rate == 0 or training:
        return 0.0
    if sampling_schedule == 'constant':
        if not 0 <= sampling_rate <= 1:
            raise ValueError(f'`constant` sampling rate must be in the interval [0, 1]. Got {sampling_schedule}.')
        sampling_probability = sampling_rate
    elif sampling_schedule == 'inverse_sigmoid':
        if sampling_rate < 1:
            raise ValueError(f'`inverse_sigmoid` sampling rate must be at least 1. Got {sampling_schedule}.')
        sampling_probability = 1.0 - sampling_rate / (sampling_rate + keras.ops.exp(step / sampling_rate))
    elif sampling_schedule == 'exponential':
        if not 0 < sampling_rate < 1:
            raise ValueError(f'`exponential` sampling rate must be in the interval (0, 1). Got {sampling_schedule}.')
        sampling_probability = 1.0 - keras.ops.power(sampling_rate, step)
    else:
        raise ValueError(f'Invalid `sampling_schedule`: {sampling_schedule}')
    return sampling_probability
