"""
This file tests features common to transformer architectures, such as
- Changing input resolution while transferring weights
- Changing input resolution during inference
"""
import numpy as np
import pytest

from tfimm.models.factory import create_model, transfer_weights

MODEL_LIST = ["vit_tiny_patch16_224", "deit_tiny_patch16_224", "cait_xxs24_224"]


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_transform_pos_embed(model_name):
    """
    We test if we can transfer weights between ViT models with different input sizes,
    which requires interpolation of position embeddings during weight transfer. This
    should be done via the transfer_weight hook in the config.
    """
    src_model = create_model(model_name)
    dst_model = create_model(model_name, input_size=(256, 256))
    transfer_weights(src_model, dst_model)

    img = dst_model.dummy_inputs
    dst_model(img)


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_interpolate_input(model_name):
    """
    We test if we can run inference with different input sizes by interpolating
    position embeddings during inference.
    """
    model = create_model(model_name)
    flexible_model = create_model(model_name, interpolate_input=True)
    transfer_weights(model, flexible_model)

    # First we test if setting `interpolate_input=True` doesn't change the output for
    # original input size.
    rng = np.random.default_rng(2021)
    img = rng.random(
        size=(1, *model.cfg.input_size, model.cfg.in_chans), dtype="float32"
    )
    res_1 = model(img)
    res_2 = flexible_model(img)
    assert (np.max(np.abs(res_1 - res_2))) / (np.max(np.abs(res_1)) + 1e-6) < 1e-6

    # Then we test, if we can run inference on input at different resolution
    img = rng.random(size=(1, 256, 256, model.cfg.in_chans), dtype="float32")
    flexible_model(img)
