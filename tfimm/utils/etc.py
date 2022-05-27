import collections.abc
from typing import Iterable, Optional, Tuple, TypeVar, Union

T = TypeVar("T")


def to_2tuple(x: Union[T, Iterable[T]]) -> Tuple[T, T]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)[:2]
    else:
        return x, x


def make_divisible(
    value: float,
    divisor: int,
    min_value: Optional[int] = None,
    round_limit: float = 0.9,
) -> int:
    """Ensures that `value` is divisible by `divisor`."""
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < round_limit * value:
        new_value += divisor
    return new_value
