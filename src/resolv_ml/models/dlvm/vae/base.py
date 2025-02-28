# TODO - DOC
from typing import Dict

import keras

from ....utilities.distributions.inference import Inference
from ....utilities.regularizers.base import Regularizer


class VAE(keras.Model):

    def __init__(self,
                 input_processing_layer: keras.Layer,
                 inference_layer: Inference,
                 sampling_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 aux_input_processing_layer: keras.Layer = None,
                 regularizers: Dict[str, Regularizer] = None,
                 name: str = "vae",
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self._input_processing_layer = input_processing_layer
        self._inference_layer = inference_layer
        self._sampling_layer = sampling_layer
        self._generative_layer = generative_layer
        self._aux_input_processing_layer = aux_input_processing_layer
        self._regularizers = regularizers
        self._evaluation_mode = False
        self._call_symbolic_build = False

    def get_latent_space_shape(self):
        raise NotImplementedError("Subclasses must implement the get_latent_space_shape method.")

    def build(self, input_shape):
        super().build(input_shape)
        vae_input_shape, aux_input_shape = input_shape
        if len(aux_input_shape) == 1:
            aux_input_shape = tuple(aux_input_shape) + (1,)
        if not self._input_processing_layer.built:
            self._input_processing_layer.build(vae_input_shape)
        if self._aux_input_processing_layer and not self._aux_input_processing_layer.built:
            self._aux_input_processing_layer.build(aux_input_shape)
        feature_extraction_out_shape = self._input_processing_layer.compute_output_shape(vae_input_shape)
        aux_input_processing_out_shape = aux_input_shape if not self._aux_input_processing_layer else \
            self._aux_input_processing_layer.compute_output_shape(aux_input_shape)
        if not self._inference_layer.built:
            self._inference_layer.build(feature_extraction_out_shape)
        sampling_in_shape = vae_input_shape, aux_input_processing_out_shape, feature_extraction_out_shape
        if not self._sampling_layer.built:
            self._sampling_layer.build(sampling_in_shape)
        sampling_out_shape = self._sampling_layer.compute_output_shape(sampling_in_shape)
        generative_in_shape = vae_input_shape, aux_input_shape, sampling_out_shape
        if not self._generative_layer.built:
            self._generative_layer.build(generative_in_shape)
        for layer in self._regularizers.values():
            if not layer.built:
                layer.build(generative_in_shape)

    def call(self, inputs, training: bool = False):
        evaluate = self._evaluation_mode or self._call_symbolic_build
        if training or evaluate:
            vae_input, aux_input = inputs
            processed_aux_input = self._aux_input_processing_layer(aux_input, training=training) \
                if self._aux_input_processing_layer else aux_input
            current_step = self.optimizer.iterations + 1
            input_features, z, posterior_dist, prior_dist = self.encode(inputs=(vae_input, processed_aux_input),
                                                                        training=training,
                                                                        evaluate=evaluate)
            outputs = self.decode(inputs=(vae_input, processed_aux_input, z),
                                  training=training,
                                  evaluate=evaluate,
                                  current_step=current_step)
            if self._regularizers:
                for regularizer_id, regularizer in self._regularizers.items():
                    regularizer_inputs = vae_input, processed_aux_input, input_features, z, outputs
                    reg_loss = regularizer(regularizer_inputs,
                                           prior=prior_dist,
                                           posterior=posterior_dist,
                                           training=training,
                                           evaluate=evaluate,
                                           current_step=current_step)
                    self.add_loss(reg_loss)
            return outputs
        else:
            if len(inputs[0].shape) == 0:
                # First input is a scalar that defines the number of samples (generation mode)
                z = self._sampling_layer(inputs=inputs[0],
                                         prior=self._inference_layer.prior_distribution())
                outputs = self.decode((z, *inputs[1:])) if len(inputs) > 1 else self.decode(z)
                return outputs, z
            else:
                # Inputs are tensors (encoding-decoding mode)
                vae_input, aux_input = inputs
                processed_aux_input = self._aux_input_processing_layer(aux_input, training=training) \
                    if self._aux_input_processing_layer else aux_input
                _, z, _, _ = self.encode(inputs=(vae_input, processed_aux_input), evaluate=True)
                outputs = self.decode(inputs=(z, vae_input))
                return outputs, z, vae_input, aux_input, processed_aux_input

    def encode(self, inputs, training: bool = False, evaluate: bool = False):
        vae_input, aux_input = inputs
        input_features = self._input_processing_layer((vae_input, aux_input), training=training)
        distributions = self._inference_layer((input_features, aux_input), training=training)
        z = self._sampling_layer((vae_input, aux_input, input_features),
                                 prior=distributions[1],
                                 posterior=distributions[0],
                                 training=training,
                                 evaluate=evaluate,
                                 symbolic_build=self._call_symbolic_build)
        return input_features, z, distributions[0], distributions[1]

    def get_latent_codes(self, n, training: bool = False, evaluate: bool = False):
        z = self._sampling_layer(n,
                                 prior=self._inference_layer.prior_distribution(training=training),
                                 training=training,
                                 evaluate=evaluate)
        return z

    def decode(self, inputs, current_step=1, training: bool = False, evaluate: bool = False):
        if training or evaluate:
            vae_input, aux_input, z = inputs
            return self._generative_layer(inputs=(vae_input, aux_input, z),
                                          training=training,
                                          current_step=keras.ops.convert_to_tensor(current_step),
                                          evaluate=evaluate)
        else:
            return self._generative_layer(inputs=inputs, training=training)

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

    def _symbolic_build(self, iterator=None, data_batch=None):
        self._call_symbolic_build = True
        super()._symbolic_build(data_batch=data_batch)
        self._call_symbolic_build = False

    def build_graph(self, input_shape):
        seq_input_shape, aux_input_shape = input_shape
        vae_input = keras.Input(shape=seq_input_shape[1:], batch_size=seq_input_shape[0], name="vae_input")
        vae_aux_input = keras.Input(shape=(None,), batch_size=aux_input_shape[0], name="vae_aux_input")
        _, z, _, _ = self.encode((vae_input, vae_aux_input), evaluate=True)
        dec_outputs = self.decode((vae_input, vae_aux_input, z), evaluate=True)
        return keras.Model(inputs=(vae_input, vae_aux_input), outputs=dec_outputs)
