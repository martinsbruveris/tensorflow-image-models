import tempfile
from pathlib import Path

import pytest
import tensorflow as tf
import tf_keras

from tfimm import create_model
from tfimm.utils.cache import get_dir, list_cached_models, set_dir, set_model_cache


@pytest.mark.parametrize("cache_type", ["dir", "model"])
def test_model_cache(cache_type):
    model_name = "test_model"
    model = create_model(model_name)

    cache_dir = tempfile.TemporaryDirectory()
    model_dir = tempfile.TemporaryDirectory()

    # Set cache directory to empty directory
    set_dir(cache_dir.name)
    assert get_dir() == cache_dir.name

    # Check model cache is empty
    assert len(list_cached_models()) == 0

    # Add model to model cache
    if cache_type == "dir":
        model.save(Path(cache_dir.name) / model_name)
    elif cache_type == "model":
        model.save(model_dir.name)
        set_model_cache(model_name, model_dir.name)
        assert list_cached_models() == [model_name]
    else:
        raise ValueError()

    # Create model from cache
    model_from_cache = create_model(model_name, pretrained=True)
    check_models_equal(model, model_from_cache)

    cache_dir.cleanup()
    model_dir.cleanup()


def check_models_equal(model_a: tf_keras.Model, model_b: tf_keras.Model):
    for var_a, var_b in zip(model_a.variables, model_b.variables):
        tf.debugging.assert_equal(var_a, var_b)
