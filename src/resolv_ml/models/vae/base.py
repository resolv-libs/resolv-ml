# TODO - DOC
from abc import ABC, abstractmethod
from typing import Any, Tuple

import keras

from resolv_ml.utilities import training as training_utils


class VAEEncoder(ABC, keras.Layer):

    def __init__(self, name: str = "vae/encoder", **kwargs):
        super(VAEEncoder, self).__init__(name=name, **kwargs)

    @abstractmethod
    def _encode(self, inputs, training: bool = False, **kwargs):
        pass

    def call(self, *args, **kwargs) -> Any:
        return self._encode(args["inputs"], args["training"], **kwargs)


class VAEDecoder(ABC, keras.Layer):

    def __init__(self, name: str = "vae/decoder", **kwargs):
        super(VAEDecoder, self).__init__(name=name, **kwargs)

    @abstractmethod
    def _decode(self, input_sequence, z, use_teacher_force, training: bool = False, **kwargs):
        pass

    def call(self, inputs, training: bool = False, **kwargs):
        input_sequence, z, use_teacher_force = inputs
        return self._decode(input_sequence, z, use_teacher_force, training=training, **kwargs)


class LatentSpace(keras.Model):

    def __init__(self,
                 z_size: int,
                 free_bits: float = None,
                 max_beta: float = None,
                 beta_rate: float = None,
                 name: str = "vae/latent_space",
                 **kwargs):
        super(LatentSpace, self).__init__(name=name, **kwargs)
        self._z_size = z_size
        self._free_bits = free_bits
        self._max_beta = max_beta
        self._beta_rate = beta_rate

    @property
    def metrics(self):
        return [
            self.kl_loss_tracker,
            self.kl_bits_tracker,
            self.kl_beta_tracker
        ]

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: Tuple[int, ...]):
        self._dense_mean = keras.layers.Dense(
            units=self._z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name=f"{self.name}/z_mean"
        )
        self._dense_log_var = keras.layers.Dense(
            units=self._z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            bias_initializer="zeros",
            name=f"{self.name}/z_log_var"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_bits_tracker = keras.metrics.Mean(name="kl_bits")
        self.kl_beta_tracker = keras.metrics.Mean(name="kl_beta")

    def call(self, inputs: Any, training: bool = False, **kwargs):
        z_mean = self._dense_mean(inputs)
        z_log_var = self._dense_log_var(inputs)
        self._add_kl_divergence_loss(z_mean, z_log_var)
        return z_mean, z_log_var

    def _add_kl_divergence_loss(self, z_mean, z_log_var):
        # Compute KL divergence for all sequences in the batch
        kl_div = -0.5 * (1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var))
        kl_div = keras.ops.sum.reduce_sum(kl_div, axis=1)
        # Compute the cost according to free_bits
        free_nats = self._free_bits * keras.ops.log(2.0)
        kl_cost = keras.ops.maximum(kl_div - free_nats, 0)
        # Compute beta for beta-VAE
        kl_beta = (1.0 - keras.ops.power(self._beta_rate, self.optimizer.iterations)) * self._max_beta
        # Compute KL across the batch and update trackers
        kl_loss = kl_beta * keras.ops.mean(kl_cost)
        kl_loss_bits = kl_loss / keras.ops.log(2.0)
        self.add_loss(kl_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_bits_tracker.update_state(kl_loss_bits)
        self.kl_beta_tracker.update_state(kl_beta)


class Sampling(keras.Layer):

    def __init__(self, name: str = "vae/sampling", **kwargs):
        super(Sampling, self).__init__(name=name, **kwargs)

    def call(self, inputs, training: bool = False, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=keras.ops.shape(z_mean), mean=0.0, stddev=1.0)
        z = z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon
        return z


class LatentCodeProcessingLayer(ABC, keras.Model):

    def __init__(self, name: str = "vae/latent_code_processing", **kwargs):
        super(LatentCodeProcessingLayer, self).__init__(name=name, **kwargs)

    @abstractmethod
    def _process_latent_code(self, latent_code, model_inputs, training: bool = False, **kwargs):
        pass

    def call(self, inputs, training: bool = False, **kwargs):
        z, model_inputs = inputs
        return self._process_latent_code(z, model_inputs, training)


class VAE(keras.Model):

    def __init__(self,
                 encoder: VAEEncoder,
                 decoder: VAEDecoder,
                 z_size: int,
                 input_layer: keras.Layer = None,
                 output_layer: keras.Layer = None,
                 z_processing_layer: LatentCodeProcessingLayer = None,
                 free_bits: float = None,
                 max_beta: float = None,
                 beta_rate: float = None,
                 sampling_schedule: str = "constant",
                 sampling_rate: float = 0.0,
                 name: str = "vae",
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.z_processing_layer = z_processing_layer
        self.latent_space_layer = LatentSpace(z_size, free_bits, max_beta, beta_rate)
        self.sampling_layer = Sampling()
        self._sampling_schedule = sampling_schedule
        self._sampling_rate = sampling_rate

    def call(self, inputs, training: bool = False, **kwargs):
        # Process inputs if necessary
        encoder_inputs = self.input_layer(inputs, training=training, **kwargs) if self.input_layer else inputs
        # Encode inputs
        encoder_outputs = self.encoder(encoder_inputs, training=training, **kwargs)
        # Compute latent space distribution
        z_mean, z_log_var = self.latent_space_layer(encoder_outputs, training=training)
        # Sample latent space
        sampling_inputs = z_mean, z_log_var
        z = self.sampling_layer(sampling_inputs, training=training)
        # Process latent code if necessary
        if self.z_processing_layer:
            z_processing_inputs = z, inputs
            z = self.z_processing_layer(z_processing_inputs, training=training, **kwargs)
        # Decode latent code
        decoder_inputs = inputs, z, self._get_teacher_force_probability(training)
        decoder_outputs = self.decoder(decoder_inputs, training=training, **kwargs)
        # Process decoder outputs if necessary
        outputs = self.output_layer(decoder_outputs, training=training,
                                    **kwargs) if self.output_layer else decoder_outputs
        return outputs

    def _get_teacher_force_probability(self, training: bool = False):
        return training_utils.get_sampling_probability(
            sampling_schedule=self._sampling_schedule,
            sampling_rate=self._sampling_rate,
            step=self.optimizer.iterations,
            training=training
        )
