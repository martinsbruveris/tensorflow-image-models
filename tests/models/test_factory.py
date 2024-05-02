import tempfile

import numpy as np
import pytest
import tensorflow as tf
import tf_keras

from tfimm.models.factory import create_model, create_preprocessing, transfer_weights

from .architectures import TEST_ARCHITECTURES  # noqa: F401

# Models for which we cannot change the input size during model creation. Examples
# are some MLP models, where the number of patches becomes the number of filters
# for convolutional kernels.
FIXED_SIZE_MODELS_CREATION = [
    "mixer_test_model",  # mlp_mixer.py
    "resmlp_test_model",
    "gmlp_test_model",
]
# Models for which we cannot change the input size during inference.
FIXED_SIZE_MODELS_INFERENCE = [
    "mixer_test_model",  # mlp_mixer.py
    "resmlp_test_model",
    "gmlp_test_model",
    # For Swin Transformers the input size influences the attention windows, which are
    # determined at build time. Hence we can change the input size at creation time,
    # but not during inference.
    "swin_test_model",  # swin.py
]
FLEXIBLE_MODELS_CREATION = list(
    set(TEST_ARCHITECTURES) - set(FIXED_SIZE_MODELS_CREATION)
)
FLEXIBLE_MODELS_INFERENCE = list(
    set(TEST_ARCHITECTURES) - set(FIXED_SIZE_MODELS_INFERENCE)
)


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
@pytest.mark.parametrize("nb_classes", [10, 0])
def test_transfer_weights(model_name, nb_classes):
    # Create two models with same architecture, but different classifiers
    model_1 = create_model(model_name)
    model_2 = create_model(model_name, nb_classes=nb_classes)

    # Transfer weights from one to another
    transfer_weights(model_1, model_2)

    img = np.random.rand(1, *model_1.cfg.input_size, model_1.cfg.in_channels)
    y_1 = model_1.forward_features(img).numpy()
    y_2 = model_2.forward_features(img).numpy()

    # We expect features to be the same for both models
    assert (np.max(np.abs(y_1 - y_2))) < 1e-6


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
@pytest.mark.parametrize("in_channels", [1, 6])
def test_change_in_channels(model_name, in_channels):
    # Create two models with same architecture, but different number of in_channels
    model_1 = create_model(model_name)
    model_2 = create_model(model_name, in_channels=in_channels)

    # Transfer weights from one to another
    transfer_weights(model_1, model_2)

    if in_channels == 1:
        # For single channel input adaptation is equivalent to replicating image
        # across the 3 channels.
        img_2 = np.random.rand(1, *model_1.cfg.input_size, 1)
        img_1 = np.repeat(img_2, repeats=3, axis=-1)
    elif in_channels == 6:
        # For six channel images adaptation is equivalent to replicating the three
        # channel image across six channels.
        img_1 = np.random.rand(1, *model_1.cfg.input_size, 3)
        img_2 = np.tile(img_1, [1, 1, 1, 2])
    else:
        raise ValueError()

    y_1 = model_1(img_1).numpy()
    y_2 = model_2(img_2).numpy()

    if model_name not in {
        "resnetv2_test_model",
        "vit_r_test_model_1",
        "vit_r_test_model_2",
    }:
        # We expect results to be the same for both models. The exception are models
        # based on ResNetV2, because they use `StdConv`, which normalizes weight
        # statistics internally. The models are still adaptable, but results won't be
        # the same.
        assert np.all(np.isclose(y_1, y_2, rtol=1e-5, atol=1e-4))


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
def test_save_load_model(model_name):
    """Tests ability to use keras save() and load() functions."""
    model = create_model(model_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = tf_keras.models.load_model(tmpdir, compile=False)

    assert type(model) is type(loaded_model)

    img = np.random.rand(1, *model.cfg.input_size, model.cfg.in_channels)
    y_1 = model(img).numpy()
    y_2 = loaded_model(img).numpy()

    assert (np.max(np.abs(y_1 - y_2))) < 1e-6


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
def test_model_path(model_name):
    """Tests ability to use `model_path` parameter in `create_model`."""
    model = create_model(model_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = create_model(model_name, model_path=tmpdir)

    assert type(model) is type(loaded_model)

    img = np.random.rand(1, *model.cfg.input_size, model.cfg.in_channels)
    y_1 = model(img).numpy()
    y_2 = loaded_model(img).numpy()

    assert (np.max(np.abs(y_1 - y_2))) < 1e-6


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
@pytest.mark.parametrize("input_size", [(8, 8), (1, 4, 4)])
@pytest.mark.parametrize("in_channels", [1, 3, 5, 6])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_preprocessing(model_name, input_size, in_channels, dtype):
    input_shape = (*input_size, in_channels)
    img = tf.ones(input_shape, dtype)
    preprocess = create_preprocessing(model_name, in_channels=in_channels, dtype=dtype)
    img = preprocess(img)
    assert img.shape == input_shape
    assert img.dtype == dtype


@pytest.mark.parametrize("model_name", FLEXIBLE_MODELS_CREATION)
def test_change_input_size(model_name):
    """
    We test if we can transfer weights between models with different input sizes,
    which requires interpolation of position embeddings during weight transfer. This
    should be done via the transfer_weight hook in the config.
    """
    src_model = create_model(model_name)
    input_size = (64, 64)
    dst_model = create_model(model_name, input_size=input_size)
    transfer_weights(src_model, dst_model)

    img = dst_model.dummy_inputs
    dst_model(img)


@pytest.mark.parametrize("model_name", FLEXIBLE_MODELS_INFERENCE)
def test_change_input_size_inference(model_name):
    """
    We test if we can run inference with different input sizes.
    """
    model = create_model(model_name)
    # For transformer models we need specify the `interpolate_input` parameter. Models
    # that don't have the parameter will ignore it.
    flexible_model = create_model(model_name, interpolate_input=True)
    transfer_weights(model, flexible_model)

    # First we test if setting `interpolate_input=True` doesn't change the output for
    # original input size.
    rng = np.random.default_rng(2021)
    img = rng.random(
        size=(1, *model.cfg.input_size, model.cfg.in_channels), dtype="float32"
    )
    res_1 = model(img)
    res_2 = flexible_model(img)
    assert (np.max(np.abs(res_1 - res_2))) / (np.max(np.abs(res_1)) + 1e-6) < 1e-6

    # Then we test, if we can run inference on input at different resolution
    img = rng.random(size=(1, 64, 64, model.cfg.in_channels), dtype="float32")
    flexible_model(img)


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
def test_model_name_keras(model_name):
    """
    We test if model.name == model.cfg.name, i.e., the keras model name is set
    correctly.
    """
    tf_keras.backend.clear_session()
    model = create_model(model_name)
    assert model.name == model_name == model.cfg.name


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
def test_variable_prefix(model_name):
    """
    We test if all model variables are created under the correct prefix
    """
    tf_keras.backend.clear_session()
    model = create_model(model_name, name="test")

    for var in model.variables:
        assert var.name.startswith("test/")


@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
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


@pytest.mark.skip()
@pytest.mark.parametrize("model_name", TEST_ARCHITECTURES)
def test_mixed_precision(model_name: str):
    """
    Test if we can run a forward pass with mixed precision.

    These tests are very slow on CPUs, so we skip them by default.
    """
    tf_keras.backend.clear_session()
    tf_keras.mixed_precision.set_global_policy("mixed_float16")
    model = create_model(model_name)
    img = tf.ones((1, *model.cfg.input_size, model.cfg.in_channels), dtype="float16")
    res = model(img)
    assert res.dtype == "float16"
