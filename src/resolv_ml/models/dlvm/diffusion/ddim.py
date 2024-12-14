
import keras


class DDIM(keras.Model):

    def __init__(self,
                 sampling_layer: keras.Layer,
                 name: str = "vae",
                 **kwargs):
        super(DDIM, self).__init__(name=name, **kwargs)
        self._sampling_layer = sampling_layer
        self._evaluation_mode = False




