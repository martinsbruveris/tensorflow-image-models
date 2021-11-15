import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import timm
import torch

from tfimm import create_model, list_models
from tfimm.utils.timm import load_pytorch_weights_in_tf2_model


# We test one (small) model from each architecture. Even then tests take time, so we
# increase the timeout.
@pytest.mark.parametrize("model_name", ["resnet18", "vit_tiny_patch16_224"])
@pytest.mark.timeout(60)
def test_save_load_model(model_name):
    model = create_model(model_name)

    # Save model and load it again
    with tempfile.TemporaryDirectory() as tmpdir:
        # We can't use h5 format for subclassed models, only saved model format.
        model_path = Path(tmpdir) / "model"
        model.save(model_path)
        loaded_model = tf.keras.models.load_model(model_path)

    # Run inference on original and loaded models
    img = model.dummy_inputs
    res = model(img).numpy()
    res_2 = loaded_model(img).numpy()
    assert np.allclose(res, res_2)


@pytest.mark.parametrize("model_name", list_models(pretrained="timm"))
@pytest.mark.timeout(60)
def test_load_timm_model(model_name):
    """Test if we can load models from timm."""
    # We don't need to load the pretrained weights from timm, we only need a PyTorch
    # model, that we then convert to tensorflow. This allows us to run these tests
    # in GitHub CI without data transfer issues.
    pt_model = timm.create_model(model_name, pretrained=False)
    pt_model.eval()

    tf_model = create_model(model_name, pretrained=False)
    tf_model = load_pytorch_weights_in_tf2_model(tf_model, pt_model.state_dict())

    rng = np.random.default_rng(2021)
    img = rng.random(
        size=(1, *tf_model.cfg.input_size, tf_model.cfg.in_chans), dtype="float32"
    )
    tf_res = tf_model(img, training=False).numpy()

    pt_img = torch.Tensor(img.transpose([0, 3, 1, 2]))
    pt_res = pt_model.forward(pt_img).detach().numpy()

    assert (np.max(np.abs(tf_res - pt_res))) < 1e-3
