# TODO - DOC
from abc import ABC, abstractmethod
from typing import Any, Tuple

import keras

from resolv_ml.utilities import training as training_utils


class VAEEncoder(ABC, keras.Layer):

    def call(self, *args, **kwargs) -> Any:
        return self._encode(args["inputs"], args["training"], **kwargs)

    @abstractmethod
    def _encode(self, inputs, training: bool = False, **kwargs) -> Any:
        pass


class VAEDecoder(ABC, keras.Layer):

    def call(self, inputs,  training: bool = False, **kwargs):
        input_sequence, z, use_teacher_force = inputs
        return self._decode(input_sequence, z, use_teacher_force, training=training, **kwargs)

    @abstractmethod
    def _decode(self, input_sequence, z, use_teacher_force, training: bool = False, **kwargs):
        pass


class LatentSpaceDistribution(keras.Layer):

    def __init__(self, z_size: int, name: str = "latent_space_distribution", **kwargs):
        super(LatentSpaceDistribution, self).__init__(name=name, **kwargs)
        self._z_size = z_size
        self._dense_log_var = None
        self._dense_mean = None

    def build(self, input_shape: Tuple[int, ...]):
        self._dense_mean = keras.layers.Dense(
            units=self._z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            name="z_mean"
        )
        self._dense_log_var = keras.layers.Dense(
            units=self._z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            bias_initializer="zeros",
            name="z_log_var"
        )

    def call(self, inputs: Any, training: bool = False, **kwargs):
        z_mean = self._dense_mean(inputs)
        z_log_var = self._dense_log_var(inputs)
        return z_mean, z_log_var


class LatentSpaceSampling(keras.Layer):
    def __init__(self, name: str = "latent_space_sampling", **kwargs):
        super(LatentSpaceSampling, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=keras.ops.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):

    def __init__(self,
                 encoder: VAEEncoder,
                 decoder: VAEDecoder,
                 z_size: int,
                 free_bits: float = None,
                 max_beta: float = None,
                 beta_rate: float = None,
                 sampling_schedule: str = "constant",
                 sampling_rate: float = 0.0,
                 name: str = "vae", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._z_size = z_size
        self._free_bits = free_bits
        self._max_beta = max_beta
        self._beta_rate = beta_rate
        self._sampling_schedule = sampling_schedule
        self._sampling_rate = sampling_rate
        self.kl_beta_tracker = None
        self.kl_bits_tracker = None
        self.kl_loss_tracker = None
        self.reconstruction_loss_tracker = None
        self.total_loss_tracker = None
        self.sampling_layer = None
        self.latent_space_layer = None

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_bits_tracker,
            self.kl_beta_tracker
        ]

    def build(self, input_shape: Tuple[int, ...]):
        self.latent_space_layer = LatentSpaceDistribution(self._z_size)
        self.sampling_layer = LatentSpaceSampling()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_bits_tracker = keras.metrics.Mean(name="kl_bits")
        self.kl_beta_tracker = keras.metrics.Mean(name="kl_beta")

    def call(self, inputs, training: bool = False, **kwargs):
        encoded_data = self.encoder(inputs, training=training)
        z_mean, z_log_var = self.latent_space_layer(encoded_data, training=training)
        z = self.sampling_layer(z_mean, z_log_var, training=training)
        reconstruction = self.decoder((inputs, z, self._get_teacher_force_probability(training)), training=training)
        self._add_kl_divergence_loss(z_mean, z_log_var)
        return reconstruction

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

    def _get_teacher_force_probability(self, training: bool = False):
        return training_utils.get_sampling_probability(
            sampling_schedule=self._sampling_schedule,
            sampling_rate=self._sampling_rate,
            step=self.optimizer.iterations,
            training=training
        )
