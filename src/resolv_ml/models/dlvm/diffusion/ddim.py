# TODO - DOC
from typing import Any

import keras

from .base import DiffusionModel


@keras.saving.register_keras_serializable(package="Diffusion", name="DDIM")
class DDIM(DiffusionModel):

    def __init__(self,
                 z_shape,
                 denoiser: keras.Layer,
                 timesteps: int,
                 loss_fn: Any = "MeanSquaredError",
                 noise_schedule_type: str = "linear",
                 noise_schedule_start: float = 1e-4,
                 noise_schedule_end: float = 0.02,
                 noise_level_conditioning: bool = False,
                 eta: float = 0.,
                 sampling_timesteps: int = 100,
                 timesteps_scheduler_type: str = 'uniform',
                 name: str = "ddim",
                 **kwargs):
        super(DDIM, self).__init__(
            z_shape=z_shape,
            denoiser=denoiser,
            timesteps=timesteps,
            loss_fn=loss_fn,
            noise_schedule_type=noise_schedule_type,
            noise_schedule_start=noise_schedule_start,
            noise_schedule_end=noise_schedule_end,
            noise_level_conditioning=noise_level_conditioning,
            name=name,
            **kwargs
        )
        self._eta = eta
        self._sampling_timesteps = sampling_timesteps
        self._timesteps_scheduler_type = timesteps_scheduler_type

    def sample(self, noisy_input):
        denoised_inputs = []
        predicted_noise = []
        x_t = noisy_input
        ddim_timesteps = self._get_ddim_timesteps()
        for timestep, prev_timestep in ddim_timesteps:
            pred_noise = self.predict_noise(x_t, timestep=timestep, training=False)
            x_t = self.denoise(x_t, pred_noise, timestep=timestep, prev_timestep=prev_timestep)
            denoised_inputs.append(x_t)
            predicted_noise.append(pred_noise)
        return keras.ops.stack(denoised_inputs, axis=1), keras.ops.stack(predicted_noise, axis=1)

    def denoise(self, noisy_input, pred_noise, timestep: int, prev_timestep: int):
        batch_size, input_shape = noisy_input.shape[0], noisy_input.shape[1:]
        noise = keras.random.normal(shape=keras.ops.shape(noisy_input)) \
            if timestep else keras.ops.zeros_like(noisy_input)
        sqrt_alpha_cumprod = self._noise_scheduler.get_tensor(
            "sqrt_alpha_cumprod", timestep, batch_size, input_shape
        )
        sqrt_one_minus_alpha_cumprod = self._noise_scheduler.get_tensor(
            "sqrt_one_minus_alpha_cumprod", timestep, batch_size, input_shape
        )
        sqrt_alpha_cumprod_prev = self._noise_scheduler.get_tensor(
            "sqrt_alpha_cumprod", prev_timestep, batch_size, input_shape
        )
        one_minus_alpha_cumprod_prev = self._noise_scheduler.get_tensor(
            "one_minus_alpha_cumprod", prev_timestep, batch_size, input_shape, first_tensor_type="zeros"
        )
        std = self._get_posterior_std(noisy_input, timestep, prev_timestep)
        pred_x_0 = (noisy_input - sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha_cumprod
        dir_xt = keras.ops.sqrt(one_minus_alpha_cumprod_prev - std ** 2)
        denoised_signal = sqrt_alpha_cumprod_prev * pred_x_0 + dir_xt * pred_noise + noise * std
        return denoised_signal

    def _get_ddim_timesteps(self):
        if self._timesteps_scheduler_type == "uniform":
            skip = self._timesteps // self._sampling_timesteps
            ddim_timesteps = list(range(0, self._timesteps, skip))
        elif self._timesteps_scheduler_type == "quad":
            seq = keras.ops.linspace(0, keras.ops.sqrt(self._sampling_timesteps * 0.8), self._timesteps) ** 2
            ddim_timesteps = [int(s) for s in list(seq)]
        else:
            raise ValueError(f"Unknown timesteps scheduler type: {self._timesteps_scheduler_type}")
        prev_timestep = [-1] + ddim_timesteps[:-1]
        return zip(reversed(ddim_timesteps), reversed(prev_timestep))

    def _get_posterior_std(self, noisy_input, timestep: int, prev_timestep: int):
        batch_size, input_shape = noisy_input.shape[0], noisy_input.shape[1:]
        if not self._eta:
            return keras.ops.zeros_like(noisy_input)
        # Compute coefficient c1 = [sqrt(1 - alpha_t_1) / sqrt(1 - alpha_t)]
        sqrt_one_minus_alpha_cumprod = self._noise_scheduler.get_tensor(
            "sqrt_one_minus_alpha_cumprod", timestep=timestep
        )
        sqrt_one_minus_alpha_cumprod_prev = self._noise_scheduler.get_tensor(
            "sqrt_one_minus_alpha_cumprod", timestep=prev_timestep, first_tensor_type="zeros"
        )
        c1 = sqrt_one_minus_alpha_cumprod_prev / sqrt_one_minus_alpha_cumprod
        # Compute coefficient c2 = [1 - (alpha_t / alpha_t_1)]
        alpha_cumprod = self._noise_scheduler.get_tensor("alpha_cumprod", timestep=timestep)
        alpha_cumprod_prev = self._noise_scheduler.get_tensor("alpha_cumprod", timestep=prev_timestep)
        c2 = keras.ops.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)
        # Compute posterior variance
        posterior_variance = self._eta * c1 * c2
        return self._noise_scheduler.get_tensor(posterior_variance, batch_size=batch_size, input_shape=input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "eta": self._eta,
            "sampling_timesteps": self._sampling_timesteps,
            "timesteps_scheduler_type": self._timesteps_scheduler_type
        }
        return {**base_config, **config}
