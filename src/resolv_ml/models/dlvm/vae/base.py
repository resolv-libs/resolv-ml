# TODO - DOC
import keras


class VAE(keras.Model):

    def __init__(self,
                 input_processing_layer: keras.Layer,
                 inference_layer: keras.Layer,
                 sampling_layer: keras.Layer,
                 generative_layer: keras.Layer,
                 dist_processing_layer: keras.Layer,
                 z_processing_layer: keras.Layer = None,
                 name: str = "vae",
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self._input_processing_layer = input_processing_layer
        self._inference_layer = inference_layer
        self._dist_processing_layer = dist_processing_layer
        self._sampling_layer = sampling_layer
        self.z_processing_layer = z_processing_layer
        self._generative_layer = generative_layer

    def call(self, inputs, training: bool = False, **kwargs):
        posterior_dist_params = None
        if training:
            # Process inputs x
            processed_inputs = self._input_processing_layer(inputs, training=training, **kwargs)
            # Infer posterior distribution q(z|x) parameters
            posterior_dist_params = self._inference_layer(processed_inputs, training=training, **kwargs)
            # Process posterior distribution parameters (e.g. add loss)
            self._dist_processing_layer(posterior_dist_params, training=training, **kwargs)
        # Sample latent code from p(z) if running in inference mode or q(z|x) if training
        z = self._sampling_layer(posterior_dist_params, training=training, **kwargs)
        # Process latent code if necessary
        processed_z = z
        if self.z_processing_layer:
            z_processing_inputs = z, inputs
            processed_z = self.z_processing_layer(z_processing_inputs, training=training, **kwargs)
        # Generate from latent code
        generative_inputs = inputs, processed_z
        outputs = self._generative_layer(generative_inputs, training=training, **kwargs)
        return outputs
