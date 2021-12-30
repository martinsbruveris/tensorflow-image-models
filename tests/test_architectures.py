import os

import numpy as np
import pytest
import tensorflow as tf
import timm
import torch

from tfimm import create_model, list_models
from tfimm.utils.timm import load_pytorch_weights_in_tf2_model

# Exclude models that cause specific test failures
if "GITHUB_ACTIONS" in os.environ:  # and 'Linux' in platform.system():
    EXCLUDE_FILTERS = [
        "cait_m*",
        "ig_resnext101_32x48d",
        "resnetv2_50x3_*",
        "resnetv2_101*",
        "resnetv2_152*",
        "vit_large_*",
        "vit_huge_*",
    ]
else:
    EXCLUDE_FILTERS = ["cait_m*"]

TIMM_ARCHITECTURES = list(
    set(list_models(exclude_filters=EXCLUDE_FILTERS)) & set(timm.list_models())
)


@pytest.mark.skip()
@pytest.mark.parametrize("model_name", list_models(exclude_filters=EXCLUDE_FILTERS))
def test_mixed_precision(model_name: str):
    """
    Test if we can run a forward pass with mixed precision.

    These tests are very slow on CPUs, so we skip them by default.
    """
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = create_model(model_name)
    img = tf.ones((1, *model.cfg.input_size, model.cfg.in_chans), dtype="float16")
    res = model(img)
    assert res.dtype == "float16"


@pytest.mark.parametrize("model_name", TIMM_ARCHITECTURES)
@pytest.mark.timeout(60)
def test_load_timm_model(model_name: str):
    """Test if we can load models from timm."""
    # We don't need to load the pretrained weights from timm, we only need a PyTorch
    # model, that we then convert to tensorflow. This allows us to run these tests
    # in GitHub CI without data transfer issues.
    pt_model = timm.create_model(model_name, pretrained=False)
    pt_model.eval()

    tf_model = create_model(model_name, pretrained=False)
    load_pytorch_weights_in_tf2_model(tf_model, pt_model.state_dict())

    rng = np.random.default_rng(2021)
    img = rng.random(
        size=(1, *tf_model.cfg.input_size, tf_model.cfg.in_chans), dtype="float32"
    )
    tf_res = tf_model(img, training=False).numpy()

    pt_img = torch.Tensor(img.transpose([0, 3, 1, 2]))
    pt_res = pt_model.forward(pt_img).detach().numpy()

    if model_name.startswith("deit_") and "distilled" in model_name:
        # During inference timm distilled models return average of both heads, while
        # we return both heads
        tf_res = tf.reduce_mean(tf_res, axis=1)

    # The tests are flaky sometimes, so we use a quite high tolerance
    assert (np.max(np.abs(tf_res - pt_res))) / (np.max(np.abs(pt_res)) + 1e-6) < 1e-3


@pytest.mark.parametrize("model_name", list_models(exclude_filters=EXCLUDE_FILTERS))
@pytest.mark.timeout(60)
def test_feature_extraction(model_name: str):
    """
    Tests if we can create a model and run inference with `return_features` set to
    both `True` and `False.
    """
    model = create_model(model_name, pretrained=False)

    inputs = model.dummy_inputs
    x1, features = model(inputs, return_features=True)
    x2 = model(inputs, return_features=False)

    # Check that return value doesn't change if we also want features
    x1, x2 = x1.numpy(), x2.numpy()
    assert np.max(np.abs(x1 - x2)) < 1e-6

    # Check that features dict contains exactly the expected keys
    assert set(features.keys()) == set(model.feature_names)
