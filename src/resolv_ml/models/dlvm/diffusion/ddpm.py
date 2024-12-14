import keras

from .base import DiffusionModel
from ....utilities.tensors.ops import batch_tensor


class DDPM(DiffusionModel):

    def __init__(self,
                 z_shape,
                 denoising_layer: keras.Layer,
                 timesteps: int,
                 loss_fn: keras.losses.Loss = keras.losses.MeanSquaredError(),
                 noise_schedule_type: str = "linear",
                 noise_schedule_start: float = 1e-4,
                 noise_schedule_end: float = 0.02,
                 noise_level_conditioning: bool = False,
                 name: str = "ddpm",
                 **kwargs):
        super(DDPM, self).__init__(
            z_shape=z_shape,
            denoising_layer=denoising_layer,
            timesteps=timesteps,
            loss_fn=loss_fn,
            noise_schedule_type=noise_schedule_type,
            noise_schedule_start=noise_schedule_start,
            noise_schedule_end=noise_schedule_end,
            name=name,
            **kwargs
        )
        self._noise_level_conditioning = noise_level_conditioning

    def call(self, inputs, training: bool = False):
        if training or self._evaluation_mode:
            diffusion_input, _ = inputs
            timestep = keras.random.randint(shape=(), minval=0, maxval=self._timesteps)
            noisy_input, noise = self.forward_diffusion(diffusion_input, timestep=timestep)
            denoised_input, pred_noise = self.denoise_input(noisy_input, timestep=timestep)
            diffusion_loss = self._loss_fn(noise, pred_noise)
            self.add_loss(diffusion_loss)
            self._ddpm_loss_tracker.update_state(diffusion_loss)
            return denoised_input, noisy_input, noise, pred_noise, timestep
        else:
            if len(inputs[0].shape) == 0:
                # First input is a scalar that defines the number of samples (generation mode)
                z = keras.random.normal(shape=(inputs[0], 1, *self._z_shape))
                denoised_inputs, pred_noise = self.sample(z)
                return denoised_inputs, pred_noise, z
            else:
                # Inputs are tensors (forward+reverse diffusion mode)
                diffusion_input, _ = inputs
                noisy_inputs, noise = self.forward_diffusion(diffusion_input)
                denoised_inputs, pred_noise = self.denoise_input(noisy_inputs)
                return denoised_inputs, noisy_inputs, noise, pred_noise

    def forward_diffusion(self, x, timestep: int = -1):
        """
        Applies forward diffusion to the input data for a specific timestep or for
        all timesteps if none is specified. The forward diffusion process adds noise
        to the input based on the combination of alpha and one-minus-alpha cumulative
        products.

        :param x: Input tensor to which diffusion noise will be added.
                  It is expected to have a shape of (batch_size, *input_shape).
        :param timestep: The timestep index used to select diffusion coefficients.
                         If not provided or None, all timesteps are processed.
        :return: A tuple containing the noisy inputs tensor and the noise tensor:
                 - noisy_inputs: Tensor of shape (batch_size, timesteps, *input_shape)
                   where noise has been added to the input.
                 - noise: Random noise tensor of the same shape as the noisy inputs,
                   which was generated and applied. Shape (batch_size, timesteps, *input_shape)
        """
        batch_size, input_shape = keras.ops.shape(x)[0], keras.ops.shape(x)[1:]
        sqrt_alpha_cumprod = self._get_noise_schedule_tensor(self._sqrt_alpha_cumprod, batch_size,
                                                             input_shape, timestep)
        sqrt_one_minus_alpha_cumprod = self._get_noise_schedule_tensor(self._sqrt_one_minus_alpha_cumprod, batch_size,
                                                                       input_shape, timestep)
        x_expanded = keras.ops.cast(keras.ops.expand_dims(x, axis=1), dtype=sqrt_alpha_cumprod.dtype)
        noise = keras.random.normal(shape=(batch_size, 1 if timestep > -1 else self._timesteps, *input_shape))
        noisy_inputs = sqrt_alpha_cumprod * x_expanded + sqrt_one_minus_alpha_cumprod * noise
        return noisy_inputs, noise

    def sample(self, noisy_input):
        denoised_inputs = []
        predicted_noise = []
        x_t = noisy_input
        for timestep in range(self._timesteps - 1, -1, -1):
            x_t, pred_noise = self.denoise_input(x_t, timestep, generation=True)
            denoised_inputs.append(keras.ops.squeeze(x_t, axis=1))
            predicted_noise.append(pred_noise)
        denoised_inputs = keras.ops.stack(denoised_inputs, axis=1)
        predicted_noise = keras.ops.stack(predicted_noise, axis=1)
        return denoised_inputs, predicted_noise

    def denoise_input(self, noisy_input, timestep: int = -1, generation: bool = False):
        batch_size, timesteps, input_shape = noisy_input.shape[0], noisy_input.shape[1], noisy_input.shape[2:]
        # There must be only one noisy input associated to a certain timestep or all noisy inputs
        assert timesteps == 1 or timesteps == self._timesteps
        if timestep > -1:
            # If timestep is provided there must be only one noisy input corresponding to that timestep noise
            assert timesteps == 1
        # Build the conditioning signal for the denoising model - shape (batch_size, timesteps)
        denoiser_cond = timestep if timestep >= 0 else range(self._timesteps)
        if self._noise_level_conditioning:
            sqrt_alpha_cumprod = keras.ops.take(self._sqrt_alpha_cumprod, timestep) if timestep > -1 \
                else self._sqrt_alpha_cumprod
            denoiser_cond = sqrt_alpha_cumprod
            if timestep > -1 and not generation:
                sqrt_alpha_cumprod_t_1 = keras.ops.take(self._sqrt_alpha_cumprod, timestep - 1) if timestep > 0 else 1.0
                denoiser_cond = keras.random.uniform(minval=sqrt_alpha_cumprod_t_1, maxval=sqrt_alpha_cumprod)
        denoiser_cond = batch_tensor(denoiser_cond, batch_size)
        # Predict the noise. Shape: (batch_size, timesteps, *input_shape)
        pred_noise = self._denoising_layer((noisy_input, denoiser_cond))
        # Obtain the denoised signals
        noise = keras.random.normal(shape=keras.ops.shape(noisy_input)) if timestep != 0 \
            else keras.ops.zeros_like(noisy_input)
        if timesteps == self._timesteps:
            # Update the noise tensor to set to zero the noise added at timestep=0
            batch_indices = keras.ops.arange(batch_size, dtype='int32')
            j_indices = keras.ops.full([batch_size], 0, dtype='int32')
            update_indices = keras.ops.stack([batch_indices, j_indices], axis=1)
            updates = keras.ops.zeros([batch_size] + input_shape, dtype='float32')
            noise = keras.ops.scatter_update(noise, indices=update_indices, updates=updates)
        sqrt_alpha = self._get_noise_schedule_tensor(self._sqrt_alpha, batch_size, input_shape, timestep)
        one_minus_alpha = self._get_noise_schedule_tensor(self._one_minus_alpha, batch_size, input_shape, timestep)
        sqrt_one_minus_alpha_cumprod = self._get_noise_schedule_tensor(self._sqrt_one_minus_alpha_cumprod, batch_size,
                                                                       input_shape, timestep)
        denoised_input = (1 / sqrt_alpha) * (noisy_input - (one_minus_alpha / sqrt_one_minus_alpha_cumprod)
                                             * pred_noise) + noise * one_minus_alpha
        return denoised_input, pred_noise

    def get_config(self):
        base_config = super().get_config()
        config = {
            "noise_level_conditioning": self._noise_level_conditioning
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        denoising_layer = keras.saving.deserialize_keras_object(config.pop("denoising_layer"))
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(
            denoising_layer=denoising_layer,
            loss_fn=loss_fn,
            **config
        )
