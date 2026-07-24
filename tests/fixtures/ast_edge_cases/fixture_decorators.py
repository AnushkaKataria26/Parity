import functools

def logged(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        return fn(*a, **kw)
    return wrapper

def retry(n):
    def decorator(fn):
        return fn
    return decorator

class Service:
    @logged
    @retry(3)
    @staticmethod
    def call(x, y=1):
        return x
