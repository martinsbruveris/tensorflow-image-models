"""
This script is used when converting models from PyTorch to TF.
"""
import numpy as np
import tensorflow as tf
import timm
import torch
from torch.hub import load_state_dict_from_url  # noqa: F401

import tfimm  # noqa: F401
from tfimm.utils.timm import load_pytorch_weights_in_tf2_model  # noqa: F401

# We need to test models in both training and inference mode (BN)
training = False
nb_calls = 3

# Load PyTorch model
pt_model = timm.create_model("resnet18", pretrained=True)
# If a model is not part of the `timm` library, we can load the state dict directly
# state_dict = load_state_dict_from_url(
#     url="https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar"  # noqa: E501
# )
# pt_model.load_state_dict(state_dict)
# Note: We should not test conversion with randomly initialized models. Because most
# normalization layers are initialized with 0 mean and variance 1, often mismatching
# normalization layers can be hidden by the initialization.

# For debug purposes we may want to print variable names, so we can compare with the
# variable names in our TF model
# for k in pt_model.state_dict().keys():
#     print(k)

if not training:  # Set PyTorch model to inference mode
    pt_model.eval()

# Load TF model
tf_model = tfimm.create_model("resnet18", pretrained=True)
# If we want to load the weights from a pytorch model outside the model factory:
# load_pytorch_weights_in_tf2_model(tf_model, pt_model.state_dict())
# For debug purposes we may want to print variable names
# for w in tf_model.weights:
#     print(w.name)

# Create test input
img = np.random.rand(5, 224, 224, 3).astype("float32")

# Run inference for TF model
tf_img = tf.constant(img)
if training:  # If training we do multiple forward passes to test BN param updates
    for _ in range(nb_calls):
        _ = tf_model(tf_img, training=training)
tf_res = tf_model(tf_img, training=training)
tf_res = tf_res.numpy()
print(tf_res.shape)

# Run inference for PyTorch model
pt_img = torch.Tensor(img.transpose([0, 3, 1, 2]))
if training:
    for _ in range(nb_calls):
        _ = pt_model.forward(pt_img)
pt_res = pt_model.forward(pt_img)
pt_res = pt_res.detach().numpy()
# When we look at output of intermediate layers, we have to transpose PyTorch data
# format (NCHW) to TF data format (NHWC). We don't have to do this, if we only look
# at the final logits
# pt_res = pt_res.transpose([0, 2, 3, 1])
print(pt_res.shape)

# Compare outputs between PyTorch and Tensorflow. We should expect the relative error
# to be <1e-5. It won't be much lower, because TF and PyTorch implement BN slightly
# differently. The two formulas are mathematically, but not numerically equivalent.
print(np.max(np.abs(tf_res - pt_res)))
print(np.max(np.abs(pt_res)))

# In case we want to check actual output. Could be useful if we suspect numeric
# instabilities, etc.
# print(np.sort(tf_res)[0, -10:])
# print(np.sort(pt_res)[0, -10:])
