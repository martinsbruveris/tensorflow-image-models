# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch - TF 2.0 general utilities."""

import logging
import re
from enum import Enum

import numpy
import tensorflow as tf
from tensorflow.python.keras import backend as K

logger = logging.getLogger(__name__)


class TransposeType(Enum):
    """
    Possible transposition types
    """

    NO = "no"
    SIMPLE = "simple"
    CONV2D = "conv2d"


def convert_tf_weight_name_to_pt_weight_name(tf_name, tf_weight_shape=None):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in
            TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0
            lists in PyTorch nn.ModulesList)
        - '/remove/' is replaced by '/' (can be used to remove additional intermediate
            levels in TF2.0)

    return tuple with:

        - pytorch model weight name
        - transpose: `TransposeType` member indicating whether and how TF2.0 and PyTorch weights matrices should be
          transposed with regards to each other
    """
    tf_name = tf_name.replace(":0", "")  # device ids
    tf_name = re.sub(
        r"/[^/]*___([^/]*)/", r"/\1/", tf_name
    )  # '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
    tf_name = tf_name.replace(
        "_._", "/"
    )  # '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
    tf_name = tf_name.replace("/remove/", "/")
    tf_name = re.sub(r"//+", "/", tf_name)  # Remove empty levels at the end
    tf_name = tf_name.split(
        "/"
    )  # Convert from TF2.0 '/' separators to PyTorch '.' separators
    # Some weights have a single name without "/" such as final_logits_bias in BART
    if len(tf_name) > 1:
        tf_name = tf_name[1:]  # Remove level zero

    # When should we transpose the weights
    if (
        tf_name[-1] in {"kernel", "depthwise_kernel"}
        and tf_weight_shape is not None
        and tf_weight_shape.rank == 4
    ):
        # A simple heuristic to detect conv layer using weight array shape
        transpose = TransposeType.CONV2D
    elif bool(
        tf_name[-1] in ["kernel", "pointwise_kernel", "depthwise_kernel"]
        or "emb_projs" in tf_name
        or "out_projs" in tf_name
    ):
        transpose = TransposeType.SIMPLE
    else:
        transpose = TransposeType.NO

    # Convert standard TF2.0 names in PyTorch names
    if tf_name[-1] in {"kernel", "depthwise_kernel", "embeddings", "gamma"}:
        tf_name[-1] = "weight"
    if tf_name[-1] == "beta":
        tf_name[-1] = "bias"

    # BatchNorm layers
    if tf_name[-1] == "moving_mean":
        tf_name[-1] = "running_mean"
    if tf_name[-1] == "moving_variance":
        tf_name[-1] = "running_var"

    # Put name together
    tf_name = ".".join(tf_name)

    return tf_name, transpose


def load_pytorch_weights_in_tf2_model(
    tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False
):
    """Load pytorch state_dict in a TF 2.0 model."""
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure model is built

    # Adapt pt state dict. TF "beta" -> PT "bias"
    # But some models have PT weight "beta" (ResMLP affine layer)
    # To fix that we need to change PT name to "bias" first...
    old_keys = []
    new_keys = []
    for key in pt_state_dict.keys():
        new_key = None
        if key.endswith(".beta"):
            new_key = key.replace(".beta", ".bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    tf_loaded_numel = 0
    weight_value_tuples = []
    all_pytorch_weights = set(list(pt_state_dict.keys()))
    missing_keys = []
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name,
            tf_weight_shape=symbolic_weight.shape,
        )

        # Find associated numpy array in pytorch model state dict
        if name not in pt_state_dict:
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            keys_to_ignore = getattr(tf_model, "keys_to_ignore_on_load_missing", None)
            if keys_to_ignore is not None:
                # authorized missing keys don't have to be loaded
                if any(re.search(pat, name) is not None for pat in keys_to_ignore):
                    continue
            print(name, sw_name)
            raise AttributeError(f"{name} not found in PyTorch model")

        array = pt_state_dict[name].numpy()

        if transpose is TransposeType.CONV2D:
            # Conv2D weight:
            #    PT: (num_out_channel, num_in_channel, kernel[0], kernel[1])
            # -> TF: (kernel[0], kernel[1], num_in_channel, num_out_channel)
            array = numpy.transpose(array, axes=(2, 3, 1, 0))
        elif transpose is TransposeType.SIMPLE:
            array = numpy.transpose(array)

        if len(symbolic_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(symbolic_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)

        if list(symbolic_weight.shape) != list(array.shape):
            try:
                array = numpy.reshape(array, symbolic_weight.shape)
            except AssertionError as e:
                e.args += (symbolic_weight.shape, array.shape)
                raise e

        try:
            assert list(symbolic_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += (symbolic_weight.shape, array.shape)
            raise e

        tf_loaded_numel += array.size

        weight_value_tuples.append((symbolic_weight, array))
        all_pytorch_weights.discard(name)

    K.batch_set_value(weight_value_tuples)

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure restore ops are run

    logger.info(f"Loaded {tf_loaded_numel:,} parameters in the TF 2.0 model.")

    unexpected_keys = list(all_pytorch_weights)
    # PyTorch BN layers track number of batches, but TF does not, so these weights
    # will always be left over.
    unexpected_keys = [
        key for key in unexpected_keys if "num_batches_tracked" not in key
    ]

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the PyTorch model were not used when initializing the "
            f"TF 2.0 model {tf_model.__class__.__name__}: {unexpected_keys}."
        )
    else:
        logger.warning(
            f"All PyTorch model weights were used when initializing "
            f"{tf_model.__class__.__name__}."
        )
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights or buffers of the TF 2.0 model "
            f"{tf_model.__class__.__name__} were not initialized from the PyTorch "
            f"model and are newly initialized: {missing_keys}."
        )
    else:
        logger.warning(
            f"All the weights of {tf_model.__class__.__name__} were initialized from "
            f"the PyTorch model.\n"
        )


def load_timm_weights(model: tf.keras.Model, model_name: str):
    """Loads weights from timm in TF model."""
    try:
        import timm
    except ImportError:
        logging.error("To load weights from timm, timm needs to be installed.")
        raise

    if not timm.is_model_pretrained(model_name):
        raise ValueError(
            f"timm model {model_name} is not pretrained. Can't load weights from timm."
        )

    pt_model = timm.create_model(model_name, pretrained=True)
    pt_state_dict = pt_model.state_dict()
    load_pytorch_weights_in_tf2_model(model, pt_state_dict)


def load_pth_url_weights(model: tf.keras.Model, url: str):
    """Loads weights from a PyTorch .pth file specified by URL."""
    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        logging.error("To load weights from pth weights, torch needs to be installed.")
        raise

    pt_state_dict = load_state_dict_from_url(url, progress=False, map_location="cpu")
    load_pytorch_weights_in_tf2_model(model, pt_state_dict)
