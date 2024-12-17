# TODO - DOC
from typing import Any

import keras

from .base import DiffusionModel


@keras.saving.register_keras_serializable(package="Diffusion", name="DDPM")
class DDPM(DiffusionModel):

    def __init__(self,
                 z_shape,
                 denoiser: keras.Layer,
                 timesteps: int,
                 loss_fn: Any = "MeanSquaredError",
                 noise_schedule_type: str = "linear",
                 noise_schedule_start: float = 1e-4,
                 noise_schedule_end: float = 0.02,
                 noise_level_conditioning: bool = False,
                 posterior_variance_type: str = "lower_bound",
                 name: str = "ddpm",
                 **kwargs):
        super(DDPM, self).__init__(
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
        # Use log variance for numerical stability
        self._posterior_variance_type = posterior_variance_type
        if self._posterior_variance_type == "upper_bound":
            self._posterior_log_variance = keras.ops.log(self._noise_scheduler.get_tensor("alpha"))
        elif self._posterior_variance_type == "lower_bound":
            one_minus_alpha = self._noise_scheduler.get_tensor("one_minus_alpha")
            one_minus_alpha_cumprod = self._noise_scheduler.get_tensor("one_minus_alpha_cumprod")
            one_minus_alpha_cumprod_prev = keras.ops.concatenate(
                [keras.ops.convert_to_tensor([0.]), one_minus_alpha_cumprod[:-1]]
            )
            pairwise_cumprod_div = one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod
            self._posterior_log_variance = keras.ops.log(one_minus_alpha * pairwise_cumprod_div)
        else:
            raise ValueError(f"Unknown posterior variance type: {posterior_variance_type}")

    def sample(self, noisy_input):
        denoised_inputs = []
        predicted_noise = []
        x_t = noisy_input
        for timestep in range(self._timesteps - 1, -1, -1):
            pred_noise = self.predict_noise(x_t, timestep=timestep, training=False)
            x_t = self.denoise(x_t, pred_noise, timestep=timestep)
            denoised_inputs.append(x_t)
            predicted_noise.append(pred_noise)
        return keras.ops.stack(denoised_inputs, axis=1), keras.ops.stack(predicted_noise, axis=1)

    def denoise(self, noisy_input, pred_noise, timestep: int):
        batch_size, input_shape = noisy_input.shape[0], noisy_input.shape[1:]
        noise = keras.random.normal(shape=keras.ops.shape(noisy_input)) \
            if timestep else keras.ops.zeros_like(noisy_input)
        rec_sqrt_alpha = self._noise_scheduler.get_tensor(
            "rec_sqrt_alpha", timestep, batch_size, input_shape
        )
        div_one_minus_alpha_sqrt_cumprod = self._noise_scheduler.get_tensor(
            "div_one_minus_alpha_sqrt_cumprod", timestep, batch_size, input_shape
        )
        posterior_log_variance = self._noise_scheduler.get_tensor(
            self._posterior_log_variance, timestep, batch_size, input_shape
        )
        std = keras.ops.exp(0.5 * posterior_log_variance)
        pred_x_t_1 = noisy_input - div_one_minus_alpha_sqrt_cumprod * pred_noise
        denoised_input = rec_sqrt_alpha * pred_x_t_1 + noise * std
        return denoised_input

    def get_config(self):
        base_config = super().get_config()
        config = {
            "posterior_variance_type": self._posterior_variance_type
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        denoiser = keras.saving.deserialize_keras_object(config.pop("denoiser"))
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(
            denoiser=denoiser,
            loss_fn=loss_fn,
            **config
        )
