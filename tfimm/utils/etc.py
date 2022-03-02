import collections.abc
from typing import Iterable, Optional, Tuple, TypeVar, Union

T = TypeVar("T")


def to_2tuple(x: Union[T, Iterable[T]]) -> Tuple[T, T]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)[:2]
    else:
        return x, x


def make_divisible(
    v: int, divisor: int, min_value: Optional[int] = None, round_limit: float = 0.9
) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
