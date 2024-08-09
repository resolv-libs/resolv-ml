import inspect

import keras
from tensorflow_probability import bijectors as tfb


@keras.saving.register_keras_serializable(package="Bijectors", name="Bijector")
class Bijector(tfb.Bijector, keras.Layer):

    def __init__(self, **kwargs):
        def filter_kwargs(method, kwargs_dict):
            signature = inspect.signature(method)
            parameters = signature.parameters
            argument_names = {k: v for k, v in kwargs_dict.items() if k in parameters}
            return argument_names

        tfb.Bijector.__init__(self, **filter_kwargs(tfb.Bijector.__init__, kwargs))
        keras.Layer.__init__(self, **filter_kwargs(keras.Layer.__init__, kwargs))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, vale):
        self._name = vale

    def get_config(self):
        config = {
            "validate_args": self._validate_args,
            "name": self._name
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
