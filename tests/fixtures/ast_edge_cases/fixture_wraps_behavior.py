import functools

def with_wraps(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

def without_wraps(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

@with_wraps
def documented_func(a, b=5):
    """Real docstring."""
    return a + b

@without_wraps
def undocumented_wrapper_func(a, b=5):
    """Real docstring."""
    return a + b
