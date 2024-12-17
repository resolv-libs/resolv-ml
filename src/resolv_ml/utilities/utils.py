import inspect


def filter_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in sig.parameters
    }
    return filtered_kwargs
