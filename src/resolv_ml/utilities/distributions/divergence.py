import keras
import keras.ops as k_ops


@keras.saving.register_keras_serializable(package="Divergences", name="BetaDivergence")
class BetaDivergenceRegularizer(keras.Layer):

    def __init__(self,
                 divergence_layer: keras.Layer,
                 max_beta: float = 1.0,
                 beta_rate: float = 0.0,
                 free_bits: float = 0.0,
                 name: str = "beta_div",
                 **kwargs):
        super(BetaDivergenceRegularizer, self).__init__(name=name, **kwargs)
        self._divergence_layer = divergence_layer
        self._free_bits = free_bits
        self._max_beta = max_beta
        self._beta_rate = beta_rate

    def build(self, input_shape):
        super().build(input_shape)
        self._divergence_layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        _, _, posterior_dist_params, _, _ = inputs
        div = self._divergence_layer(posterior_dist_params, training=training)
        # Compute the cost according to free_bits
        free_nats = self._free_bits * k_ops.log(2.0)
        div_cost = k_ops.maximum(div - free_nats, 0)
        # Compute beta for beta-VAE
        div_beta = (1.0 - k_ops.power(self._beta_rate, kwargs.get("iterations", 0))) * self._max_beta
        # Compute KL across the batch and update trackers
        div_loss = div_beta * k_ops.mean(div_cost)
        div_loss_bits = div_loss / k_ops.log(2.0)
        return div_loss, div_loss_bits, div_beta

    def compute_output_shape(self, input_shape):
        return (3,)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "divergence_layer": keras.saving.serialize_keras_object(self._divergence_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        divergence_layer = keras.saving.deserialize_keras_object(config.pop("divergence_layer"))
        return cls(divergence_layer, **config)


@keras.saving.register_keras_serializable(package="Divergences", name="GaussianKLD")
class GaussianKLDivergence(keras.Layer):

    def __init__(self, name: str = "gauss_kl_div", **kwargs):
        super(GaussianKLDivergence, self).__init__(name=name, **kwargs)

    def call(self, inputs, training: bool = False, **kwargs):
        p_mean, p_var, q_mean, q_var = tuple(inputs) + (0, 1)
        kl_loss = k_ops.sum(0.5 * (k_ops.log(p_var/q_var) + (k_ops.square(q_mean-p_mean) + q_var)/p_var - 1), axis=-1)
        return k_ops.mean(kl_loss)

    def compute_output_shape(self, input_shape):
        return (1,)
