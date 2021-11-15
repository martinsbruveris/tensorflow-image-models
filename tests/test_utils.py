import pytest

from tfimm.utils import to_2tuple


@pytest.mark.parametrize(
    "x, y",
    [
        (1, (1, 1)),  # Scalar input
        ((2, 2), (2, 2)),  # Tuple input, don't change
        ((3, 3, 3), (3, 3)),  # Too long tuple, cut
        (range(4), (0, 1)),  # Iterable input
        (["a", "b"], ("a", "b")),  # Different dtype
    ],
)
def test_to_2tuple(x, y):
    assert to_2tuple(x) == y
