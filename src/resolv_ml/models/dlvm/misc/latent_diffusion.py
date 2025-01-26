# TODO - DOC
import keras

from ..diffusion.base import DiffusionModel
from ..vae.base import VAE


class LatentDiffusion(keras.Model):

    def __init__(self,
                 vae: VAE | str,
                 diffusion: DiffusionModel,
                 name: str = "latent_diffusion",
                 **kwargs):
        super(LatentDiffusion, self).__init__(name=name, **kwargs)
        if isinstance(vae, str):
            vae = keras.models.load_model(vae, compile=False)
        elif not isinstance(vae, VAE):
            raise ValueError(f"Invalid VAE type: {type(vae)}. Expected a VAE or a path to a saved VAE model.")
        self._vae = vae
        self._diffusion = diffusion
        self._vae.trainable = False
        self._evaluation_mode = False
        self._diff_loss_tracker = keras.metrics.Mean(name=f"{diffusion.name}_loss")

    def build(self, input_shape):
        if not self._vae.built:
            self._vae.build(input_shape)
        if not self._diffusion.built:
            vae_input_shape, cond_input_shape = input_shape
            vae_latent_space_shape = self._vae.get_latent_space_shape()
            self._diffusion.build((vae_latent_space_shape, cond_input_shape))

    def call(self, inputs, training: bool = None):
        if training or self._evaluation_mode:
            vae_input, cond_input = inputs
            _, z, _, _ = self._vae.encode((vae_input, cond_input), training=training, evaluate=self._evaluation_mode)
            noise, pred_noise, timestep, loss = self._diffusion((z, cond_input), training=training)
            self._diff_loss_tracker.update_state(loss)
            return noise, pred_noise, timestep
        else:
            if len(inputs) < 3:
                n_samples, *decoder_inputs = inputs
                z_labels = None
            else:
                n_samples, z_labels, *decoder_inputs = inputs
            z, pred_noise, _ = self._diffusion((n_samples, z_labels), training=training)
            output = self._vae.decode(
                inputs=(z[:, -1, ...], *decoder_inputs), training=training, evaluate=self._evaluation_mode
            )
            return output, pred_noise, z

    def sample(self, inputs):
        z_noisy, *decoder_inputs = inputs
        z_denoised, pred_noise = self._diffusion.sample(z_noisy)
        output = self._vae.decode(inputs=(z_denoised[:, -1, ...], *decoder_inputs))
        return output, pred_noise, z_denoised

    def get_latent_codes(self, n):
        return self._diffusion.get_latent_codes(n)

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
        self._diffusion._evaluation_mode = True
        self._vae._evaluation_mode = True
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
        self._diffusion._evaluation_mode = False
        self._vae._evaluation_mode = False
        return eval_output

    def get_config(self):
        base_config = super().get_config()
        config = {
            "vae": keras.saving.serialize_keras_object(self._vae),
            "diffusion": keras.saving.serialize_keras_object(self._diffusion)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        vae = keras.saving.deserialize_keras_object(config.pop("vae"))
        diffusion = keras.saving.deserialize_keras_object(config.pop("diffusion"))
        return cls(
            vae=vae,
            diffusion=diffusion,
            **config
        )
