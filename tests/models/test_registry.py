import pytest

from tfimm.models.registry import _split_model_name, _update_cfg_url


@pytest.mark.parametrize(
    "full_name, model_name, model_tag",
    [("a.b", "a", "b"), ("a", "a", ""), ("a.b.c", "a", "b.c")],
)
def test_split_model_name(full_name, model_name, model_tag):
    assert _split_model_name(full_name) == (model_name, model_tag)


@pytest.mark.parametrize(
    "url, model_name, model_tag, expected",
    [
        ("[timm]pt", "tf", "tag", "[timm]pt.tag"),
        ("[timm]pt.old", "tf", "tag", "[timm]pt.tag"),
        ("[timm]", "tf", "tag", "[timm]tf.tag"),
        ("[not-timm]", "tf", "tag", "[not-timm]"),
    ],
)
def test_update_cfg_url(url, model_name, model_tag, expected):
    assert _update_cfg_url(url, model_name, model_tag) == expected
