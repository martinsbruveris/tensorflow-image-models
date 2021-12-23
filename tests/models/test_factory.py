import tempfile

import numpy as np
import pytest
import tensorflow as tf

from tfimm import list_models
from tfimm.models.factory import create_model, create_preprocessing, transfer_weigths

MODEL_LIST = [
    "convmixer_768_32",  # convmixer.py
    "mixer_s32_224",  # mlp_mixer.py
    "resmlp_12_224",
    "gmlp_ti16_224",
    "pvt_tiny",  # pvt.py
    "pvt_v2_b0",   # pvt_v2.py
    "resnet18",  # resnet.py
    "swin_tiny_patch4_window7_224",  # swin.py
    "deit_tiny_distilled_patch16_224",  # vit.py
    "vit_tiny_patch16_224",
]


@pytest.mark.parametrize("model_name", MODEL_LIST)
@pytest.mark.parametrize("nb_classes", [10, 0])
def test_transfer_weights(model_name, nb_classes):
    # Create two models with same architecture, but different classifiers
    model_1 = create_model(model_name)
    model_2 = create_model(model_name, nb_classes=nb_classes)

    # Transfer weights from one to another
    transfer_weigths(model_1, model_2)

    img = np.random.rand(1, *model_1.cfg.input_size, model_1.cfg.in_chans)
    y_1 = model_1.forward_features(img).numpy()
    y_2 = model_2.forward_features(img).numpy()

    # We expect features to be the same for both models
    assert (np.max(np.abs(y_1 - y_2))) < 1e-6


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_save_load_model(model_name):
    """Tests ability to use keras save() and load() functions."""
    model = create_model(model_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = tf.keras.models.load_model(tmpdir)

    assert type(model) is type(loaded_model)

    img = np.random.rand(1, *model.cfg.input_size, model.cfg.in_chans)
    y_1 = model(img).numpy()
    y_2 = loaded_model(img).numpy()

    assert (np.max(np.abs(y_1 - y_2))) < 1e-6


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_model_path(model_name):
    """Tests ability to use `model_path` parameter in `create_model`."""
    model = create_model(model_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = create_model(model_name, model_path=tmpdir)

    assert type(model) is type(loaded_model)

    img = np.random.rand(1, *model.cfg.input_size, model.cfg.in_chans)
    y_1 = model(img).numpy()
    y_2 = loaded_model(img).numpy()

    assert (np.max(np.abs(y_1 - y_2))) < 1e-6


@pytest.mark.parametrize("model_name", list_models())
@pytest.mark.parametrize("input_shape", [(8, 8, 3), (1, 4, 4, 3)])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_preprocessing(model_name, input_shape, dtype):
    img = tf.ones(input_shape, dtype)
    preprocess = create_preprocessing(model_name, dtype)
    img = preprocess(img)
    assert img.shape == input_shape
    assert img.dtype == dtype
