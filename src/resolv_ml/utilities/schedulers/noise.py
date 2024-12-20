from typing import Tuple

import keras

from resolv_ml.utilities.ops import batch_tensor


def get_noise_schedule(timesteps: int = 1000,
                       noise_schedule_type: str = "linear",
                       noise_schedule_start: float = 1e-4,
                       noise_schedule_end: float = 0.02):
    if noise_schedule_type == "linear":
        beta = keras.ops.linspace(start=noise_schedule_start, stop=noise_schedule_end, num=timesteps)
    elif noise_schedule_type == "cosine":
        # TODO - Noise schedulers
        beta = 0
    elif noise_schedule_type == "sqrt":
        # TODO - Noise schedulers
        beta = 0
    else:
        raise ValueError(f"Unsupported noise schedule type: {noise_schedule_type}")
    return beta


class NoiseScheduler:

    def __init__(self,
                 timesteps: int = 1000,
                 noise_schedule_type: str = "linear",
                 noise_schedule_start: float = 1e-4,
                 noise_schedule_end: float = 0.02):
        beta = get_noise_schedule(timesteps, noise_schedule_type, noise_schedule_start, noise_schedule_end)
        alpha = 1.0 - beta
        sqrt_alpha = keras.ops.sqrt(alpha)
        rec_sqrt_alpha = 1 / sqrt_alpha
        one_minus_alpha = 1.0 - alpha
        alpha_cumprod = keras.ops.cumprod(alpha)
        one_minus_alpha_cumprod = 1.0 - alpha_cumprod
        sqrt_one_minus_alpha_cumprod = keras.ops.sqrt(one_minus_alpha_cumprod)
        self._schedule_tensors = {
            "beta": beta,
            "alpha": alpha,
            "sqrt_alpha": sqrt_alpha,
            "rec_sqrt_alpha": rec_sqrt_alpha,
            "one_minus_alpha": one_minus_alpha,
            "sqrt_one_minus_alpha": keras.ops.sqrt(one_minus_alpha),
            "alpha_cumprod": alpha_cumprod,
            "one_minus_alpha_cumprod": one_minus_alpha_cumprod,
            "sqrt_alpha_cumprod": keras.ops.sqrt(alpha_cumprod),
            "sqrt_one_minus_alpha_cumprod": sqrt_one_minus_alpha_cumprod,
            "div_one_minus_alpha_sqrt_cumprod": one_minus_alpha / sqrt_one_minus_alpha_cumprod
        }

    def get_tensor(self,
                   tensor,
                   timestep: int = None,
                   batch_size: int = None,
                   input_shape: Tuple[int] = None,
                   first_tensor_type: str = "ones"):
        # Timestep < 0 means that we are at step 0 of the denoising process, return a tensor of ones or zeros
        # according to the specified type
        if timestep and timestep < 0:
            if first_tensor_type == "ones":
                first_tensor = keras.ops.convert_to_tensor([1.])
            elif first_tensor_type == "zeros":
                first_tensor = keras.ops.convert_to_tensor([0.])
            else:
                raise ValueError(f"Unsupported first tensor type: {first_tensor_type}")
            return self.get_tensor(first_tensor, batch_size=batch_size, input_shape=input_shape)

        tensor = self._schedule_tensors[tensor] if isinstance(tensor, str) else tensor
        tensor = keras.ops.take(tensor, timestep) if timestep is not None else tensor
        if input_shape:
            tensor = keras.ops.reshape(tensor, newshape=[1] * len(input_shape))
        if batch_size:
            tensor = batch_tensor(tensor, batch_size)
        return tensor
