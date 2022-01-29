import numpy as np
import pytest
import tensorflow as tf
import timm
import torch

import tfimm.architectures.timm  # noqa: F401
from tfimm import create_model
from tfimm.utils.timm import load_pytorch_weights_in_tf2_model

TIMM_ARCHITECTURES = [
    "cait_xxs24_224",  # cait.py
    "convmixer_768_32",  # convmixer.py
    "convnext_tiny",  # convnext.py
    "mixer_s32_224",  # mlp_mixer.py
    "resmlp_12_224",
    "gmlp_ti16_224",
    "pit_ti_224",  # pit.py
    "pit_ti_distilled_224",
    "poolformer_s12",  # poolformer.py
    "pvt_tiny",  # pvt.py
    "pvt_v2_b0",  # pvt_v2.py
    "resnet18",  # resnet.py
    "resnetv2_50x1_bitm",  # resnetv2.py
    "swin_tiny_patch4_window7_224",  # swin.py
    "deit_tiny_patch16_224",  # vit.py
    "vit_tiny_patch16_224",
    "vit_tiny_r_s16_p8_224",  # vit_hybrid.py
    "vit_small_r26_s32_224",
]


@pytest.mark.parametrize("model_name", TIMM_ARCHITECTURES)
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
        size=(1, *tf_model.cfg.input_size, tf_model.cfg.in_channels), dtype="float32"
    )
    tf_res = tf_model(img, training=False).numpy()

    pt_img = torch.Tensor(img.transpose([0, 3, 1, 2]))
    pt_res = pt_model.forward(pt_img).detach().numpy()

    if "distilled" in model_name:
        # During inference timm distilled models return average of both heads, while
        # we return both heads
        tf_res = tf.reduce_mean(tf_res, axis=1)

    # The tests are flaky sometimes, so we use a quite high tolerance
    assert (np.max(np.abs(tf_res - pt_res))) / (np.max(np.abs(pt_res)) + 1e-6) < 1e-3
