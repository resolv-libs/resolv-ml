# TODO - DOC
import keras

from .base import VAE
from ....utilities.distributions.divergence import BetaDivergence, GaussianKLDivergence
from ....utilities.distributions.inference import GaussianInference, GaussianReparametrizationTrick


class StandardVAE(VAE):

    def __init__(self,
                 z_size: int,
                 input_processing_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 mean_inference_layer: keras.Layer = None,
                 log_var_inference_layer: keras.Layer = None,
                 z_processing_layer: keras.Layer = None,
                 free_bits: float = None,
                 max_beta: float = None,
                 beta_rate: float = None,
                 name: str = "standard_vae",
                 **kwargs):
        super(StandardVAE, self).__init__(
            input_processing_layer=input_processing_layer,
            generative_layer=generative_layer,
            inference_layer=GaussianInference(
                z_size=z_size,
                mean_inference_layer=mean_inference_layer,
                log_var_inference_layer=log_var_inference_layer,
                name=f"{name}/gaussian_inference",
            ),
            dist_processing_layer=BetaDivergence(
                divergence_layer=GaussianKLDivergence(),
                max_beta=max_beta,
                beta_rate=beta_rate,
                free_bits=free_bits
            ),
            sampling_layer=GaussianReparametrizationTrick(),
            z_processing_layer=z_processing_layer,
            name=name,
            **kwargs
        )
