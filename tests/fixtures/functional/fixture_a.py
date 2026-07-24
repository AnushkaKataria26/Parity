# fixture_a.py
import os

CONST = 1

def top_level_fn(a, b=2):
    """Top level docstring."""
    def inner_fn(x):
        return x
    return inner_fn(a)

class Outer:
    """Outer class docstring."""
    class_var = 1

    def __init__(self, x):
        self.x = x

    @staticmethod
    def static_method(y):
        return y

    @property
    def prop(self):
        return self.x

    class Inner:
        def inner_method(self):
            pass

async def async_top_level():
    pass
