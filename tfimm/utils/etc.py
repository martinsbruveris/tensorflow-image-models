import collections.abc
from typing import Iterable, Tuple, TypeVar, Union

T = TypeVar("T")


def to_2tuple(x: Union[T, Iterable[T]]) -> Tuple[T, T]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)[:2]
    else:
        return x, x
