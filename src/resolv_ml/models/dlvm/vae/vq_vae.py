# TODO - DOC
# TODO - add multi-backend support for probability distributions

import keras
from tensorflow_probability import distributions as tfd

from .base import VAE
from ....utilities.distributions.inference import CategoricalInference
from ....utilities.math.distances import sqrt_euclidean_distance
from ....utilities.regularizers.codebook import VectorQuantizationRegularizer, CommitmentRegularizer
from ....utilities.schedulers import Scheduler
from resolv_ml.utilities.ops import batch_tensor


@keras.saving.register_keras_serializable(package="VAE", name="VQ-VAE")
class VQVAE(VAE):

    def __init__(self,
                 z_size: int,
                 codebook_size: int,
                 feature_extraction_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 aux_input_processing_layer: keras.Layer = None,
                 vq_scheduler: Scheduler = None,
                 commitment_scheduler: Scheduler = None,
                 name: str = "vq_vae",
                 **kwargs):
        self._z_size = z_size
        self._codebook_size = codebook_size
        self._vq_scheduler = vq_scheduler
        self._commitment_scheduler = commitment_scheduler
        super(VQVAE, self).__init__(
            feature_extraction_layer=feature_extraction_layer,
            generative_layer=generative_layer,
            inference_layer=CategoricalInference(
                codebook_size=codebook_size,
                name="categorical_inference"
            ),
            sampling_layer=VQSamplingLayer(z_size=z_size, codebook_size=codebook_size),
            aux_input_processing_layer=aux_input_processing_layer,
            regularizers={
                "vq": VectorQuantizationRegularizer(weight_scheduler=vq_scheduler),
                "commitment": CommitmentRegularizer(weight_scheduler=commitment_scheduler)
            },
            name=name,
            **kwargs
        )

    def get_latent_space_shape(self):
        return (self._z_size,)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "z_size": self._z_size,
            "codebook_size": self._codebook_size,
            "vq_scheduler": keras.saving.serialize_keras_object(self._vq_scheduler),
            "commitment_scheduler": keras.saving.serialize_keras_object(self._commitment_scheduler),
            "feature_extraction_layer": keras.saving.serialize_keras_object(self._feature_extraction_layer),
            "generative_layer": keras.saving.serialize_keras_object(self._generative_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        feature_extraction_layer = keras.saving.deserialize_keras_object(config.pop("feature_extraction_layer"))
        generative_layer = keras.saving.deserialize_keras_object(config.pop("generative_layer"))
        vq_scheduler = keras.saving.deserialize_keras_object(config.pop("vq_scheduler"))
        commitment_scheduler = keras.saving.deserialize_keras_object(config.pop("commitment_scheduler"))
        return cls(
            feature_extraction_layer=feature_extraction_layer,
            generative_layer=generative_layer,
            vq_scheduler=vq_scheduler,
            commitment_scheduler=commitment_scheduler,
            **config
        )


@keras.saving.register_keras_serializable(package="VAE", name="CodebookSamplingLayer")
class VQSamplingLayer(keras.Layer):

    def __init__(self,
                 z_size: int,
                 codebook_size: int,
                 name: str = "vq_sampling",
                 **kwargs):
        super(VQSamplingLayer, self).__init__(name=name, **kwargs)
        self._z_size = z_size
        self._codebook_size = codebook_size
        self._codebook = keras.layers.Embedding(self._codebook_size, self._z_size)

    def build(self, input_shape):
        self._codebook.build()

    def compute_output_shape(self, input_shape, **kwargs):
        _, _, input_features_shape = input_shape
        return input_features_shape[0], *input_features_shape[1:], self._z_size

    def call(self,
             inputs,
             prior: tfd.Distribution,
             posterior: tfd.Distribution = None,
             training: bool = False,
             evaluate: bool = False,
             **kwargs):
        codebook_weights = self._codebook.get_weights()[0]
        if training or evaluate:
            # Substitute input features with the closest embeddings from codebook
            # Input features tensor's shape must have the channel dimension as the second one
            _, _, input_features = inputs
            exp_input_features = input_features
            if len(input_features.shape) < 3:
                exp_input_features = keras.ops.expand_dims(input_features, axis=-1)
            batch_size, features_dim, *spatial_dim = exp_input_features.shape
            permute_feature_tensor = keras.ops.transpose(exp_input_features,
                                                         axes=(0, *list(range(2, len(spatial_dim) + 2)), 1))
            reshaped_feature_tensor = keras.ops.reshape(permute_feature_tensor, newshape=(batch_size, -1, features_dim))
            batched_codebook = batch_tensor(codebook_weights, batch_size)
            distances = sqrt_euclidean_distance(reshaped_feature_tensor, batched_codebook)
            closest_emb_idx = keras.ops.argmin(distances, axis=-1)
            quantized = keras.ops.take(codebook_weights, closest_emb_idx, axis=0)
            quantized_reshaped = keras.ops.reshape(quantized, newshape=(batch_size, *spatial_dim, features_dim))
            permute_feature_tensor = keras.ops.transpose(
                quantized_reshaped,
                axes=(0, len(quantized_reshaped.shape) - 1, *list(range(1, len(spatial_dim) + 1)))
            )
            if len(input_features.shape) < 3:
                permute_feature_tensor = keras.ops.squeeze(permute_feature_tensor, axis=-1)
            z = permute_feature_tensor
        else:
            # TODO - allow to learn an autoregressive model to learn the true prior
            # Sample an embedding from the codebook with a uniform distribution
            n_samples = inputs
            emb_idx = prior.sample(sample_shape=(n_samples,))
            quantized = keras.ops.take(codebook_weights, emb_idx, axis=0)
            z = quantized
        return z
