# TODO - DOC
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import keras


class VAE(keras.Model, ABC):

    def __init__(self,
                 input_processing_layer: keras.Layer,
                 inference_layer: keras.Layer,
                 sampling_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 regularization_layers: List[keras.Layer] = None,
                 input_shape: tuple = (None, None),
                 aux_input_shape: tuple = (None,),
                 name: str = "vae",
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self._input_shape = input_shape
        self._aux_input_shape = aux_input_shape
        self._iterations = keras.Variable(initializer=0, name="iterations")
        # Override layers names
        input_processing_layer.name = "input_processing"
        inference_layer.name = "inference"
        sampling_layer.name = "sampling"
        generative_layer.name = "generative"
        # Build encoder model
        vae_input = keras.Input(shape=input_shape, name="input")
        aux_input = keras.Input(shape=aux_input_shape, name="aux_input")
        input_processing_layer_out = input_processing_layer(vae_input)  # frontend
        inference_layer_out = inference_layer(input_processing_layer_out)
        z = sampling_layer((inference_layer_out, aux_input))
        self._encoder = keras.Model(inputs=(vae_input, aux_input), outputs=(z, *inference_layer_out), name="encoder")
        # Build decoder model
        decoder_inputs = vae_input, aux_input, z
        generative_layer_out = generative_layer(decoder_inputs, iterations=self._iterations)
        self._decoder = keras.Model(inputs=decoder_inputs, outputs=generative_layer_out, name="decoder")
        # Add regularization layers
        self._regularization_layers = regularization_layers

    @abstractmethod
    def _add_regularization_losses(self, regularization_losses):
        pass

    def build(self, input_shape):
        super().build(input_shape)
        self._encoder.build(input_shape)
        self._decoder.build(input_shape)
        for layer in self._regularization_layers:
            layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        if training:
            vae_input, aux_input = inputs
            self._iterations.assign(self.optimizer.iterations + 1)
            z, *posterior_dist_params = self._encoder((vae_input, aux_input), training=training)
            outputs = self._decoder((vae_input, aux_input, z), training=training)
            regularization_losses = []
            if self._regularization_layers:
                for regularization_layer in self._regularization_layers:
                    regularizer_inputs = vae_input, aux_input, posterior_dist_params, z, outputs
                    layer_reg_losses = regularization_layer(regularizer_inputs, training=training,
                                                            iterations=self._iterations)
                    regularization_losses.append(layer_reg_losses)
                self._add_regularization_losses(regularization_losses)
        else:
            z = inputs
            vae_input = keras.ops.convert_to_tensor([0])
            aux_input = keras.ops.convert_to_tensor([0])
            outputs = self._decoder.predict((vae_input, aux_input, z))
        return outputs

    def plot(self, path: Union[str, Path]):
        # TODO - Fix VAE plot (don't know what the problem is)
        vae_input = keras.Input(shape=self._input_shape, name="vae_input")
        vae_aux_input = keras.Input(shape=self._aux_input_shape, name="vae_aux_input")
        z, *_ = self._encoder((vae_input, vae_aux_input))
        dec_outputs = self._decoder((vae_input, vae_aux_input, z))
        vae_model = keras.Model(inputs=(vae_input, vae_aux_input), outputs=dec_outputs)
        keras.utils.plot_model(
            vae_model,
            show_shapes=False,  # TODO - Keras bug can't print shapes for layers with multiple outputs
            show_layer_names=True,
            show_dtype=False,  # TODO - Keras bug can't print dtypes for layers with multiple outputs
            to_file=str(path) if path else "./vae_plot.png",
            expand_nested=True
        )
