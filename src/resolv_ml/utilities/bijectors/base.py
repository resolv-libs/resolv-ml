import keras
from tensorflow_probability import bijectors as tfb


@keras.saving.register_keras_serializable(package="Bijectors", name="Bijector")
class Bijector(tfb.Bijector):

    def get_config(self):
        config = {
            "validate_args": self._validate_args,
            "name": self._name
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
