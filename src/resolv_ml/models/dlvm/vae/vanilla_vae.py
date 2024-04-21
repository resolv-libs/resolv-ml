# TODO - DOC
import keras

from .base import VAE
from ....utilities.distributions.divergence import BetaDivergenceRegularizer, GaussianKLDivergence
from ....utilities.distributions.inference import GaussianInference, GaussianReparametrizationTrick


@keras.saving.register_keras_serializable(package="VAE", name="StandardVAE")
class StandardVAE(VAE):

    def __init__(self,
                 z_size: int,
                 input_processing_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 mean_inference_layer: keras.Layer = None,
                 log_var_inference_layer: keras.Layer = None,
                 max_beta: float = 1.0,
                 beta_rate: float = 0.0,
                 free_bits: float = 0.0,
                 name: str = "standard_vae",
                 **kwargs):
        self._z_size = z_size
        self._max_beta = max_beta
        self._beta_rate = beta_rate
        self._free_bits = free_bits
        self._mean_inference_layer = mean_inference_layer
        self._log_var_inference_layer = log_var_inference_layer
        self.div_loss_tracker = keras.metrics.Mean(name=f"kl_loss")
        self.div_bits_tracker = keras.metrics.Mean(name=f"kl_bits")
        self.div_beta_tracker = keras.metrics.Mean(name=f"kl_beta")
        super(StandardVAE, self).__init__(
            input_processing_layer=input_processing_layer,
            generative_layer=generative_layer,
            inference_layer=GaussianInference(
                z_size=z_size,
                mean_layer=mean_inference_layer,
                log_var_layer=log_var_inference_layer,
                name="gaussian_inference",
            ),
            sampling_layer=GaussianReparametrizationTrick(z_size=z_size),
            regularization_layers=[BetaDivergenceRegularizer(
                divergence_layer=GaussianKLDivergence(),
                max_beta=max_beta,
                beta_rate=beta_rate,
                free_bits=free_bits
            )],
            name=name,
            **kwargs
        )

    def build(self, input_shape):
        super().build(input_shape)

    def _add_regularization_losses(self, regularization_losses):
        div_loss, div_loss_bits, div_beta = regularization_losses[0]
        self.add_loss(div_loss)
        self.div_loss_tracker.update_state(div_loss)
        self.div_bits_tracker.update_state(div_loss_bits)
        self.div_beta_tracker.update_state(div_beta)
        
    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_size": self._z_size,
            "max_beta": self._max_beta,
            "beta_rate": self._beta_rate,
            "free_bits": self._free_bits,
            "input_processing_layer": keras.saving.serialize_keras_object(self._input_processing_layer),
            "mean_inference_layer": keras.saving.serialize_keras_object(self._mean_inference_layer),
            "log_var_inference_layer": keras.saving.serialize_keras_object(self._log_var_inference_layer),
            "generative_layer": keras.saving.serialize_keras_object(self._generative_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_processing_layer = keras.saving.deserialize_keras_object(config.pop("input_processing_layer"))
        mean_inference_layer = keras.saving.deserialize_keras_object(config.pop("mean_inference_layer"))
        log_var_inference_layer = keras.saving.deserialize_keras_object(config.pop("log_var_inference_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        return cls(
            input_processing_layer=input_processing_layer,
            mean_inference_layer=mean_inference_layer,
            log_var_inference_layer=log_var_inference_layer,
            generative_layer=generative_layer,
            **config
        )
