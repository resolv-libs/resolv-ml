# TODO - DOC
from typing import Tuple

import keras
from ....utilities.tensors.ops import batch_tensor


class DiffusionModel(keras.Model):

    def __init__(self,
                 z_shape: Tuple[int],
                 denoising_layer: keras.Layer,
                 timesteps: int,
                 loss_fn: keras.losses.Loss = keras.losses.MeanSquaredError(),
                 noise_schedule_type: str = "linear",
                 noise_schedule_start: float = 1e-4,
                 noise_schedule_end: float = 0.02,
                 name: str = "diffusion",
                 **kwargs):
        super(DiffusionModel, self).__init__(name=name, **kwargs)
        self._z_shape = z_shape
        self._denoising_layer = denoising_layer
        self._timesteps = timesteps
        self._loss_fn = loss_fn
        self._noise_schedule_type = noise_schedule_type
        self._noise_schedule_start = noise_schedule_start
        self._noise_schedule_end = noise_schedule_end
        self._build_noise_schedule()
        self._evaluation_mode = False
        self._ddpm_loss_tracker = keras.metrics.Mean(name=f"{name}_loss")

    def build(self, input_shape):
        super().build(input_shape)
        self._denoising_layer.build(input_shape)

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
        self._alpha_cumprod = keras.ops.cumprod(self._alpha)
        self._one_minus_alpha_cumprod = 1.0 - self._alpha_cumprod
        self._sqrt_alpha_cumprod = keras.ops.sqrt(self._alpha_cumprod)
        self._sqrt_one_minus_alpha_cumprod = keras.ops.sqrt(self._one_minus_alpha_cumprod)

    def _get_noise_schedule_tensor(self, tensor, batch_size: int, input_shape: Tuple[int], timestep: int = -1):
        tensor = keras.ops.take(tensor, timestep) if timestep > -1 else tensor
        tensor = keras.ops.reshape(tensor, newshape=[1 if timestep > -1 else self._timesteps] + [1] * len(input_shape))
        tensor = batch_tensor(tensor, batch_size)
        return tensor

    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_shape": self._z_shape,
            "denoising_layer": keras.saving.serialize_keras_object(self._denoising_layer),
            "timesteps": self._timesteps,
            "loss_fn": keras.saving.serialize_keras_object(self._loss_fn),
            "noise_schedule_type": self._noise_schedule_type,
            "noise_schedule_start": self._noise_schedule_start,
            "noise_schedule_end": self._noise_schedule_end
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
