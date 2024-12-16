# TODO - DOC
from typing import Tuple

import keras

from ....utilities.schedulers.noise import NoiseScheduler
from ....utilities.tensors.ops import batch_tensor


class DiffusionModel(keras.Model):

    def __init__(self,
                 z_shape: Tuple[int],
                 denoiser: keras.Layer,
                 timesteps: int = 1000,
                 loss_fn: keras.losses.Loss = keras.losses.MeanSquaredError(),
                 noise_schedule_type: str = "linear",
                 noise_schedule_start: float = 1e-4,
                 noise_schedule_end: float = 0.02,
                 noise_level_conditioning: bool = False,
                 name: str = "diffusion",
                 **kwargs):
        super(DiffusionModel, self).__init__(name=name, **kwargs)
        self._z_shape = z_shape
        self._denoiser = denoiser
        self._timesteps = timesteps
        self._loss_fn = loss_fn
        self._noise_schedule_type = noise_schedule_type
        self._noise_schedule_start = noise_schedule_start
        self._noise_schedule_end = noise_schedule_end
        self._noise_level_conditioning = noise_level_conditioning
        self._noise_scheduler = NoiseScheduler(
            noise_schedule_type=noise_schedule_type,
            noise_schedule_start=noise_schedule_start,
            noise_schedule_end=noise_schedule_end
        )
        self._evaluation_mode = False
        self._ddpm_loss_tracker = keras.metrics.Mean(name=f"{name}_loss")

    def sample(self, z):
        raise NotImplementedError("Subclasses must implement the sample method.")

    def build(self, input_shape):
        super().build(input_shape)
        self._denoiser.build(input_shape)

    def call(self, inputs, training: bool = False):
        if training or self._evaluation_mode:
            diffusion_input, _ = inputs
            # Monte Carlo sampling of timestep during training TODO - Antithetic sampling
            timestep = keras.random.randint(shape=(), minval=0, maxval=self._timesteps)
            noisy_input, noise = self.forward_diffusion(diffusion_input, timestep=timestep)
            pred_noise = self.predict_noise(noisy_input, timestep=timestep, training=training)
            diffusion_loss = self._loss_fn(noise, pred_noise)
            self.add_loss(diffusion_loss)
            self._ddpm_loss_tracker.update_state(diffusion_loss)
            return noise, pred_noise, timestep
        else:
            # Input is a scalar that defines the number of samples (generation mode)
            n_samples = inputs[0]
            z = keras.random.normal(shape=(n_samples, 1, *self._z_shape))
            denoised_inputs, pred_noise = self.sample(z)
            return denoised_inputs, pred_noise, z

    def forward_diffusion(self, x, timestep: int):
        """
        Applies forward diffusion to the input data for a specific timestep or for
        all timesteps if none is specified. The forward diffusion process adds noise
        to the input based on the combination of alpha and one-minus-alpha cumulative
        products.

        :param x: Input tensor to which diffusion noise will be added.
                  It is expected to have a shape of (batch_size, *input_shape).
        :param timestep: The timestep indexes used to select diffusion coefficients.
        :return: A tuple containing the noisy inputs tensor and the noise tensor:
                 - noisy_inputs: Tensor of shape (batch_size, timesteps, *input_shape)
                   where noise has been added to the input.
                 - noise: Random noise tensor of the same shape as the noisy inputs,
                   which was generated and applied. Shape (batch_size, timesteps, *input_shape)
        """
        assert 0 <= timestep < self._timesteps, f"Invalid timestep: {timestep}"
        batch_size, input_shape = keras.ops.shape(x)[0], keras.ops.shape(x)[2:]
        sqrt_alpha_cumprod = self._noise_scheduler.get_tensor("sqrt_alpha_cumprod", timestep, batch_size, input_shape)
        sqrt_one_minus_alpha_cumprod = self._noise_scheduler.get_tensor(
            "sqrt_one_minus_alpha_cumprod", timestep, batch_size, input_shape
        )
        x_expanded = keras.ops.cast(keras.ops.expand_dims(x, axis=1), dtype=sqrt_alpha_cumprod.dtype)
        noise = keras.random.normal(shape=(batch_size, 1, *input_shape))
        noisy_inputs = sqrt_alpha_cumprod * x_expanded + sqrt_one_minus_alpha_cumprod * noise
        return noisy_inputs, noise

    def predict_noise(self, noisy_input, timestep: int, training: bool = False):
        # Build the conditioning signal for the denoising model - shape (batch_size, timesteps)
        denoiser_cond = self._get_denoiser_conditioning(noisy_input, timestep=timestep, training=training)
        # Predict the noise. Shape: (batch_size, timesteps, *input_shape)
        pred_noise = self._denoiser((noisy_input, denoiser_cond), training=training)
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

    def _get_denoiser_conditioning(self, noisy_input, timestep: int, training: bool = False):
        batch_size = noisy_input.shape[0]
        denoiser_cond = timestep
        if self._noise_level_conditioning:
            sqrt_alpha_cumprod = self._noise_scheduler.get_tensor("sqrt_alpha_cumprod", timestep=timestep)
            denoiser_cond = sqrt_alpha_cumprod
            if training:
                prev_timestep = timestep - 1
                sqrt_alpha_cumprod_prev = self._noise_scheduler.get_tensor("sqrt_alpha_cumprod", timestep=prev_timestep)
                denoiser_cond = keras.random.uniform(
                    shape=(), minval=sqrt_alpha_cumprod_prev, maxval=sqrt_alpha_cumprod
                )
        denoiser_cond = batch_tensor(denoiser_cond, batch_size)
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
            "noise_level_conditioning": self._noise_level_conditioning
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
