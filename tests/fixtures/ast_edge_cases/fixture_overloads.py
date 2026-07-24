from typing import overload

@overload
def process(x: int) -> int: ...
@overload
def process(x: str) -> str: ...
def process(x):
    return x

if True:
    def conditional_fn():
        return 1
else:
    def conditional_fn():
        return 2
