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
        self._input_processing_layer = input_processing_layer
        self._inference_layer = inference_layer
        self._sampling_layer = sampling_layer
        self._generative_layer = generative_layer
        self._regularization_layers = regularization_layers
        self._evaluation_mode = False

    def _add_regularization_losses(self, regularization_losses):
        raise NotImplementedError("VAE subclasses must implement _add_regularization_losses.")

    def build(self, input_shape):
        super().build(input_shape)
        vae_input_shape, aux_input_shape = input_shape
        if not self._input_processing_layer.built:
            self._input_processing_layer.build(vae_input_shape)
        input_processing_out_shape = self._input_processing_layer.compute_output_shape(vae_input_shape)
        if not self._inference_layer.built:
            self._inference_layer.build(input_processing_out_shape)
        if not self._sampling_layer.built:
            self._sampling_layer.build(aux_input_shape)
        sampling_out_shape = self._sampling_layer.compute_output_shape(aux_input_shape)
        if not self._generative_layer.built:
            self._generative_layer.build(tuple(input_shape) + (sampling_out_shape,))
        for layer in self._regularization_layers:
            if not layer.built:
                layer.build(input_shape)

    def call(self, inputs, training: bool = False, **kwargs):
        if training or self._evaluation_mode:
            vae_input, aux_input = inputs
            iterations = self.optimizer.iterations + 1
            z, posterior_dist = self.encode(inputs=(vae_input, aux_input),
                                            training=training,
                                            evaluate=self._evaluation_mode,
                                            **kwargs)
            outputs = self.decode(inputs=(vae_input, aux_input, z),
                                  training=training,
                                  evaluate=self._evaluation_mode,
                                  iterations=iterations,
                                  **kwargs)
            regularization_losses = []
            if self._regularization_layers:
                for regularization_layer in self._regularization_layers:
                    regularizer_inputs = vae_input, aux_input, z, outputs
                    layer_reg_losses = regularization_layer(regularizer_inputs,
                                                            posterior=posterior_dist,
                                                            training=training,
                                                            evaluate=self._evaluation_mode,
                                                            iterations=iterations)
                    regularization_losses.append(layer_reg_losses)
                self._add_regularization_losses(regularization_losses)
            return outputs
        else:
            if len(inputs[0].shape) == 0 and len(inputs[1].shape) == 0:
                # Inputs are scalars (generation mode)
                num_sequences, seq_length = inputs
                z = self._sampling_layer(inputs=num_sequences)
                outputs = self.decode((z, seq_length), **kwargs)
                return outputs, z
            else:
                # Inputs are tensors (encoding-decoding mode)
                vae_input, aux_input = inputs
                sequence_length = keras.ops.convert_to_tensor(vae_input.shape[1])
                z, _ = self.encode(inputs=(vae_input, aux_input), evaluate=True, **kwargs)
                outputs = self.decode(inputs=(z, sequence_length), **kwargs)
                return outputs, z, vae_input, aux_input

    def encode(self, inputs, training: bool = False, evaluate: bool = False, **kwargs):
        vae_input, aux_input = inputs
        input_processing_layer_out = self._input_processing_layer(vae_input, training=training, **kwargs)
        posterior_dist = self._inference_layer(input_processing_layer_out, training=training, **kwargs)
        z = self._sampling_layer(aux_input,
                                 posterior=posterior_dist,
                                 training=training,
                                 evaluate=evaluate,
                                 **kwargs)
        return z, posterior_dist

    def sample(self, num_samples, training: bool = False, evaluate: bool = False, **kwargs):
        z = self._sampling_layer(num_samples, training=training, evaluate=evaluate, **kwargs)
        return z

    def decode(self, inputs, iterations=None, training: bool = False, evaluate: bool = False, **kwargs):
        if training or evaluate:
            vae_input, aux_input, z = inputs
            return self._generative_layer(inputs=(vae_input, aux_input, z),
                                          training=training,
                                          iterations=keras.ops.convert_to_tensor(iterations or 1),
                                          evaluate=keras.ops.convert_to_tensor(evaluate),
                                          **kwargs)
        else:
            z, seq_lengths = inputs
            return self._generative_layer(inputs=(z, seq_lengths), training=training, **kwargs)

    def evaluate(
            self,
            x=None,
            y=None,
            batch_size=None,
            verbose="auto",
            sample_weight=None,
            steps=None,
            callbacks=None,
            return_dict=False,
            **kwargs
    ):
        self._evaluation_mode = True
        eval_output = super().evaluate(
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks,
            return_dict=return_dict,
            **kwargs
        )
        self._evaluation_mode = False
        return eval_output

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
        z, *_ = self.encode((vae_input, vae_aux_input), evaluate=True)
        dec_outputs = self.decode((vae_input, vae_aux_input, z), evaluate=True)
        return keras.Model(inputs=(vae_input, vae_aux_input), outputs=dec_outputs)
