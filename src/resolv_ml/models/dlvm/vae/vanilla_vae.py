# TODO - DOC
import keras

from .base import VAE
from ....utilities.distributions.divergence import BetaDivergenceRegularizer, GaussianKLDivergence
from ....utilities.distributions.inference import GaussianInference, GaussianReparametrizationTrick


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
                 input_shape: tuple = (None, None),
                 aux_input_shape: tuple = (None,),
                 name: str = "standard_vae",
                 **kwargs):
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
            input_shape=input_shape,
            aux_input_shape=aux_input_shape,
            name=name,
            **kwargs
        )
        self.div_loss_tracker = keras.metrics.Mean(name=f"kl_loss")
        self.div_bits_tracker = keras.metrics.Mean(name=f"kl_bits")
        self.div_beta_tracker = keras.metrics.Mean(name=f"kl_beta")

    @property
    def metrics(self):
        return super().metrics + [
            self.div_loss_tracker,
            self.div_bits_tracker,
            self.div_beta_tracker
        ]

    def build(self, input_shape):
        super().build(input_shape)

    def _add_regularization_losses(self, regularization_losses):
        div_loss, div_loss_bits, div_beta = regularization_losses
        self.add_loss(div_loss)
        self.div_loss_tracker.update_state(div_loss)
        self.div_bits_tracker.update_state(div_loss_bits)
        self.div_beta_tracker.update_state(div_beta)
