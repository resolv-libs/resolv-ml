# TODO - DOC
from typing import Tuple, Any, Callable

import keras
import numpy as np

from ....utilities.schedulers import Scheduler
from ....utilities.schedulers.noise import NoiseScheduler
from ....utilities.ops import batch_tensor


class DiffusionModel(keras.Model):

    def __init__(self,
                 z_shape: Tuple[int],
                 denoiser: keras.Layer,
                 timesteps: int = 1000,
                 loss_fn: Any = "MeanSquaredError",
                 noise_fn: Callable = None,
                 noise_schedule_type: str = "linear",
                 noise_schedule_start: float = 1e-4,
                 noise_schedule_end: float = 0.02,
                 noise_level_conditioning: bool = False,
                 cfg_uncond_probability_scheduler: Scheduler = None,
                 cfg_weight: float = None,
                 name: str = "diffusion",
                 **kwargs):
        super(DiffusionModel, self).__init__(name=name, **kwargs)
        self._z_shape = z_shape
        self._denoiser = denoiser
        self._timesteps = timesteps
        self._loss_fn = keras.losses.get(loss_fn)
        self._noise_fn = noise_fn
        self._noise_schedule_type = noise_schedule_type
        self._noise_schedule_start = noise_schedule_start
        self._noise_schedule_end = noise_schedule_end
        self._noise_level_conditioning = noise_level_conditioning
        self._cfg_uncond_probability_scheduler = cfg_uncond_probability_scheduler
        self._cfg_weight = cfg_weight
        self._noise_scheduler = NoiseScheduler(
            noise_schedule_type=noise_schedule_type,
            noise_schedule_start=noise_schedule_start,
            noise_schedule_end=noise_schedule_end
        )
        self._evaluation_mode = False
        self._call_symbolic_build = False

    def sample(self, z, z_labels=None):
        raise NotImplementedError("Subclasses must implement the sample method.")

    def build(self, input_shape):
        diffusion_input_shape = input_shape[0]
        batch_size = diffusion_input_shape[0]
        conditioning_input_shape = batch_size, 1
        super().build(diffusion_input_shape)
        if not self._denoiser.built:
            self._denoiser.build((diffusion_input_shape, conditioning_input_shape))

    def call(self, inputs, current_step=None, training: bool = False):
        evaluate = self._evaluation_mode or self._call_symbolic_build
        if training or evaluate:
            x, x_labels = inputs
            # Monte Carlo sampling of timestep during training
            timestep = np.random.randint(low=0, high=self._timesteps)
            if self._cfg_uncond_probability_scheduler is not None:
                current_step = current_step if current_step else self.optimizer.iterations + 1
                cfg_uncond_probability = self._cfg_uncond_probability_scheduler(step=current_step)
                x_labels = x_labels if keras.random.uniform(shape=()) >= cfg_uncond_probability else None
            x_noisy, noise = self.forward_diffusion(x, x_labels, timestep=timestep)
            pred_noise = self.predict_noise(x_noisy, x_labels=x_labels, timestep=timestep, training=training)
            diffusion_loss = self._loss_fn(noise, pred_noise)
            self.add_loss(diffusion_loss)
            return noise, pred_noise, timestep, diffusion_loss
        else:
            # First input is a scalar that defines the number of samples (generation mode)
            # Second input are the conditioning labels
            n_samples, z_labels = (inputs[0][1], inputs[1]) if len(inputs) == 2 else (inputs[0], None)
            z = self.get_noise(n_samples)
            denoised_inputs, pred_noise = self.sample(z, z_labels)
            return denoised_inputs, pred_noise, z

    def get_latent_codes(self, n, labels=None):
        return self.get_noise(n, x_labels=labels)

    def get_noise(self, n: int, x=None, x_labels=None):
        return self._noise_fn(self, x=x, labels=x_labels, batch_size=n) if self._noise_fn \
            else self.get_gaussian_noise(n, x)

    def get_gaussian_noise(self, n: int, x=None):
        n = keras.ops.convert_to_tensor(n)
        return keras.random.normal(shape=(n, *self._z_shape) if x is None else (n, *x.shape[1:]))

    def forward_diffusion(self, x, x_labels, timestep: int):
        assert 0 <= timestep < self._timesteps, f"Invalid timestep: {timestep}"
        batch_size, input_shape = x.shape[0], x.shape[1:]
        sqrt_alpha_cumprod = self._noise_scheduler.get_tensor("sqrt_alpha_cumprod", timestep, batch_size, input_shape)
        sqrt_one_minus_alpha_cumprod = self._noise_scheduler.get_tensor(
            "sqrt_one_minus_alpha_cumprod", timestep, batch_size, input_shape
        )
        x = keras.ops.cast(x, dtype=sqrt_alpha_cumprod.dtype)
        noise = self.get_noise(n=batch_size, x=x, x_labels=x_labels)
        noisy_inputs = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        return noisy_inputs, noise

    def predict_noise(self, x_noisy, timestep: int, x_labels=None, training: bool = False):
        denoiser_cond = self._get_denoiser_conditioning(
            x_noisy, timestep=timestep, x_labels=x_labels, training=training
        )
        pred_noise = self._denoiser((x_noisy, x_labels, denoiser_cond), training=training)
        if self._cfg_weight:
            uncond_pred_noise = self._denoiser((x_noisy, None, denoiser_cond), training=training)
            pred_noise = (1 + self._cfg_weight) * pred_noise - self._cfg_weight * uncond_pred_noise
        return pred_noise

    def evaluate(
            self,
            x=None,
            y=None,
            batch_size=None,
            verbose="auto",
            sample_weight=None,
            steps=None,
            callbacks=None,
            return_dict=False,
            **kwargs
    ):
        self._evaluation_mode = True
        eval_output = super().evaluate(
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks,
            return_dict=return_dict,
            **kwargs
        )
        self._evaluation_mode = False
        return eval_output

    def _symbolic_build(self, iterator=None, data_batch=None):
        self._call_symbolic_build = True
        super()._symbolic_build(data_batch=data_batch)
        self._call_symbolic_build = False

    def _get_denoiser_conditioning(self, noisy_input, timestep: int, x_labels=None, training: bool = False):
        batch_size = noisy_input.shape[0]
        denoiser_t_cond = timestep
        if self._noise_level_conditioning:
            sqrt_alpha_cumprod = self._noise_scheduler.get_tensor("sqrt_alpha_cumprod", timestep=timestep)
            denoiser_t_cond = sqrt_alpha_cumprod
            if training:
                prev_timestep = timestep - 1
                sqrt_alpha_cumprod_prev = self._noise_scheduler.get_tensor("sqrt_alpha_cumprod", timestep=prev_timestep)
                denoiser_t_cond = keras.random.uniform(
                    shape=(1,), minval=sqrt_alpha_cumprod_prev, maxval=sqrt_alpha_cumprod
                )
            else:
                denoiser_t_cond = keras.ops.expand_dims(denoiser_t_cond, axis=-1)
        denoiser_t_cond = batch_tensor(denoiser_t_cond, batch_size)
        denoiser_cond = keras.ops.concatenate([denoiser_t_cond, x_labels], axis=-1) if x_labels is not None \
            else denoiser_t_cond
        return denoiser_cond

    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_shape": self._z_shape,
            "denoiser": keras.saving.serialize_keras_object(self._denoiser),
            "timesteps": self._timesteps,
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn),
            "noise_schedule_type": self._noise_schedule_type,
            "noise_schedule_start": self._noise_schedule_start,
            "noise_schedule_end": self._noise_schedule_end,
            "noise_level_conditioning": self._noise_level_conditioning,
            "cfg_uncond_probability_scheduler": keras.saving.serialize_keras_object(
                self._cfg_uncond_probability_scheduler
            ),
            "cfg_weight": self._cfg_weight
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        denoiser = keras.saving.deserialize_keras_object(config.pop("denoiser"))
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        cfg_uncond_probability_scheduler = keras.saving.deserialize_keras_object(
            config.pop("cfg_uncond_probability_scheduler")
        )
        return cls(
            denoiser=denoiser,
            loss_fn=loss_fn,
            cfg_uncond_probability_scheduler=cfg_uncond_probability_scheduler,
            **config
        )
