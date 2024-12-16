# TODO - DOC
from typing import Tuple, List

import keras
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
        self._build_noise_schedule()
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
        sqrt_alpha_cumprod = self._get_noise_schedule_tensor(self._sqrt_alpha_cumprod, timestep, batch_size,
                                                             input_shape)
        sqrt_one_minus_alpha_cumprod = self._get_noise_schedule_tensor(self._sqrt_one_minus_alpha_cumprod, timestep, batch_size,
                                                                       input_shape)
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

    def _build_noise_schedule(self):
        if self._noise_schedule_type == "linear":
            beta = keras.ops.linspace(self._noise_schedule_start, self._noise_schedule_end, self._timesteps)
            self._alpha = 1.0 - beta
        elif self._noise_schedule_type == "cosine":
            # TODO - Noise schedulers
            beta = 0
            self._alpha = 1.0 - beta
        elif self._noise_schedule_type == "sqrt":
            # TODO - Noise schedulers
            beta = 0
            self._alpha = 1.0 - beta
        else:
            raise ValueError(f"Unsupported noise schedule type: {self._noise_schedule_type}")
        self._sqrt_alpha = keras.ops.sqrt(self._alpha)
        self._one_minus_alpha = 1.0 - self._alpha
        self._sqrt_one_minus_alpha = keras.ops.sqrt(self._one_minus_alpha)
        self._alpha_cumprod = keras.ops.cumprod(self._alpha)
        self._one_minus_alpha_cumprod = 1.0 - self._alpha_cumprod
        self._sqrt_alpha_cumprod = keras.ops.sqrt(self._alpha_cumprod)
        self._sqrt_one_minus_alpha_cumprod = keras.ops.sqrt(self._one_minus_alpha_cumprod)

    def _get_noise_schedule_tensor(self,
                                   tensor,
                                   timestep: int = None,
                                   batch_size: int = None,
                                   input_shape: Tuple[int] = None):
        tensor = keras.ops.take(tensor, timestep) if timestep is not None else tensor
        if input_shape:
            tensor = keras.ops.reshape(tensor, newshape=[1] * (len(input_shape) + 1))
        if batch_size:
            tensor = batch_tensor(tensor, batch_size)
        return tensor

    def _get_prev_noise_schedule_tensor(self,
                                        tensor,
                                        timestep: int = None,
                                        batch_size: int = None,
                                        input_shape: Tuple[int] = None,
                                        first_tensor_type: str = "ones"):
        # Timestep < 0 means that we are at step 0 of the denoising process, return a tensor of ones or zeros according
        # to the specified type
        if timestep and timestep < 0:
            if first_tensor_type == "ones":
                first_tensor = keras.ops.convert_to_tensor([1.])
            elif first_tensor_type == "zeros":
                first_tensor = keras.ops.convert_to_tensor([0.])
            else:
                raise ValueError(f"Unsupported first tensor type: {first_tensor_type}")
            return self._get_prev_noise_schedule_tensor(first_tensor, batch_size=batch_size, input_shape=input_shape)

        return self._get_noise_schedule_tensor(
            tensor, timestep=timestep, batch_size=batch_size, input_shape=input_shape
        )

    def _get_denoiser_conditioning(self, noisy_input, timestep: int, training: bool = False):
        batch_size = noisy_input.shape[0]
        denoiser_cond = timestep
        if self._noise_level_conditioning:
            sqrt_alpha_cumprod = self._get_noise_schedule_tensor(self._sqrt_alpha_cumprod, timestep=timestep)
            denoiser_cond = sqrt_alpha_cumprod
            if training:
                sqrt_alpha_cumprod_prev = self._get_prev_noise_schedule_tensor(
                    self._sqrt_alpha_cumprod, timestep=timestep
                )
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
