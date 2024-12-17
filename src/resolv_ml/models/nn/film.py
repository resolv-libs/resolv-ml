# TODO - DOC
from typing import Any

import keras
from keras.src import activations


class FiLM(keras.Layer):
    """
    Applies Feature-wise Linear Modulation (FiLM) to input layers, enabling input-
    conditional modifications by coupling learned scale and shift parameters.
    Paper reference: https://arxiv.org/abs/1709.07871 by Perez et al. 2017

    FiLM is designed to facilitate conditional processing of input data by applying
    affine transformations determined by the `gamma_layer` and `beta_layer`. These
    learnable layers compute the scaling (gamma) and shifting (beta) parameters
    respectively, based on input activations. These parameters allow flexible
    feature-wise adaptation, making FiLM particularly useful in tasks like
    context-aware feature modifications in deep neural networks.

    :ivar gamma_layer: The layer responsible for computing scaling (gamma)
        parameters from the input.
    :type gamma_layer: keras.layers.Layer
    :ivar beta_layer: The layer responsible for computing shifting (beta)
        parameters from the input.
    :type beta_layer: keras.layers.Layer
    """
    def __init__(self,
                 gamma_layer: keras.Layer = keras.layers.Dense(2048),
                 beta_layer: keras.Layer = keras.layers.Dense(2048),
                 name="film",
                 **kwargs):
        super(FiLM, self).__init__(name=name, **kwargs)
        self.gamma_layer = gamma_layer
        self.beta_layer = beta_layer

    def call(self, x, training: bool = False):
        scale = self.gamma_layer(x)
        shift = self.beta_layer(x)
        return scale, shift

    def get_config(self):
        base_config = super().get_config()
        config = {
            "gamma_layer": keras.saving.serialize_keras_object(self.gamma_layer),
            "beta_layer": keras.saving.serialize_keras_object(self.beta_layer)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        gamma_layer = keras.saving.deserialize_keras_object(config.pop("gamma_layer"))
        beta_layer = keras.saving.deserialize_keras_object(config.pop("beta_layer"))
        return cls(
            gamma_layer=gamma_layer,
            beta_layer=beta_layer,
            **config
        )


class FeatureWiseAffine(keras.layers.Layer):
    """
    Applies feature-wise affine transformation to the input tensor. This layer modifies
    the input by scaling and shifting, controlled by feature-wise parameters provided
    as inputs. It is typically used in normalization operations or adaptive processing
    of intermediate features in machine learning models.

    :ivar name: Name of the layer instance.
    :type name: str
    """
    def __init__(self,
                 activation: Any = None,
                 name="feature_wise_affine",
                 **kwargs):
        super(FeatureWiseAffine, self).__init__(name=name, **kwargs)
        self.activation = activations.get(activation)

    def call(self, inputs, training: bool = False):
        x, scale, shift = inputs
        for _ in range(len(x.shape) - 2):
            scale = keras.ops.expand_dims(scale, axis=1)
            shift = keras.ops.expand_dims(shift, axis=1)
        output = scale * x + shift
        if self.activation:
            output = self.activation(output)
        return output
