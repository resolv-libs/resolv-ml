# TODO - DOC
from typing import List

import keras


class VAE(keras.Model):

    def __init__(self,
                 input_processing_layer: keras.Layer,
                 inference_layer: keras.Layer,
                 sampling_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 regularization_layers: List[keras.Layer] = None,
                 name: str = "vae",
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        # Encoder layers
        input_processing_layer.name = "input_processing"
        inference_layer.name = "inference"
        sampling_layer.name = "sampling"
        self._input_processing_layer = input_processing_layer
        self._inference_layer = inference_layer
        self._sampling_layer = sampling_layer
        # Decoder layers
        generative_layer.name = "generative"
        self._generative_layer = generative_layer
        # Add regularization layers
        self._regularization_layers = regularization_layers

    def _add_regularization_losses(self, regularization_losses):
        raise NotImplementedError("VAE subclasses must implement _add_regularization_losses.")

    def build(self, input_shape):
        super().build(input_shape)
        vae_input_shape, aux_input_shape = input_shape
        self._input_processing_layer.build(vae_input_shape)
        input_processing_out_shape = self._input_processing_layer.compute_output_shape(vae_input_shape)
        self._inference_layer.build(input_processing_out_shape)
        inference_out_shape = self._inference_layer.compute_output_shape(input_processing_out_shape)
        self._sampling_layer.build((*inference_out_shape, aux_input_shape))
        sampling_out_shape = self._sampling_layer.compute_output_shape((*inference_out_shape, aux_input_shape))
        self._generative_layer.build(tuple(input_shape) + (sampling_out_shape,))
        for layer in self._regularization_layers:
            layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        if training:
            vae_input, aux_input = inputs
            iterations = self.optimizer.iterations + 1
            z, *posterior_dist_params = self.encode(inputs, training=training, **kwargs)
            outputs = self.decode((vae_input, aux_input, z), training=training, iterations=iterations)
            regularization_losses = []
            if self._regularization_layers:
                for regularization_layer in self._regularization_layers:
                    regularizer_inputs = vae_input, aux_input, posterior_dist_params, z, outputs
                    layer_reg_losses = regularization_layer(regularizer_inputs, training=training,
                                                            iterations=iterations)
                    regularization_losses.append(layer_reg_losses)
                self._add_regularization_losses(regularization_losses)
        else:
            z = inputs
            outputs = self.decode(z, training=training, **kwargs)
        return outputs

    def encode(self, inputs, training: bool = False, **kwargs):
        vae_input, aux_input = inputs
        input_processing_layer_out = self._input_processing_layer(vae_input, training=training, **kwargs)
        posterior_dist_params = self._inference_layer(input_processing_layer_out, training=training, **kwargs)
        z = self._sampling_layer((*posterior_dist_params, aux_input), training=training, **kwargs)
        return z, *posterior_dist_params

    def decode(self, inputs, training: bool = False, **kwargs):
        if training:
            vae_input, aux_input, z = inputs
            return self._generative_layer((vae_input, aux_input, z), training=training,
                                          iterations=kwargs.pop('iterations', 1), **kwargs)
        else:
            z = inputs
            return self._generative_layer(z, training=training, **kwargs)

    def print_summary(
            self,
            input_shape,
            line_length=None,
            positions=None,
            print_fn=None,
            expand_nested=False,
            show_trainable=False,
            layer_range=None
    ):
        graph = self.build_graph(input_shape)
        graph.summary(line_length, positions, print_fn, expand_nested, show_trainable, layer_range)

    def build_graph(self, input_shape):
        seq_input_shape, aux_input_shape = input_shape
        vae_input = keras.Input(shape=seq_input_shape[1:], batch_size=seq_input_shape[0], name="vae_input")
        vae_aux_input = keras.Input(shape=aux_input_shape[1:], batch_size=aux_input_shape[0], name="vae_aux_input")
        z, *_ = self.encode((vae_input, vae_aux_input), training=True)
        dec_outputs = self.decode((vae_input, vae_aux_input, z), training=True)
        return keras.Model(inputs=(vae_input, vae_aux_input), outputs=dec_outputs)
