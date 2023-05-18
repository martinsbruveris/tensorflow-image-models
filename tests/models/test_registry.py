import pytest

from tfimm.models.registry import _split_model_name


@pytest.mark.parametrize(
    "full_name, model_name, model_tag",
    [("a.b", "a", "b"), ("a", "a", ""), ("a.b.c", "a", "b.c")],
)
def test_split_model_name(full_name, model_name, model_tag):
    assert _split_model_name(full_name) == (model_name, model_tag)
