import keras


def batch_tensor(tensor, batch_size):
    expanded_tensor = keras.ops.expand_dims(tensor, axis=0)
    batched_tensor = keras.ops.repeat(expanded_tensor, repeats=batch_size, axis=0)
    return batched_tensor
