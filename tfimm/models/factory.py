import logging
import os
import re
from copy import deepcopy
from typing import Callable, Optional, Union

import tensorflow as tf
from tensorflow.python.keras import backend as K

from tfimm.models.registry import is_model, model_class, model_config
from tfimm.utils import get_dir, load_pth_url_weights, load_timm_weights


# TODO: Implement in_channels, to work with both timm as well as saved models
def create_model(
    model_name: str,
    pretrained: Union[bool, str] = False,
    model_path: str = "",
    *,
    in_channels: Optional[int] = None,
    nb_classes: Optional[int] = None,
    **kwargs,
):
    """Creates a model.

    Args:
        model_name: Name of model to instantiate
        pretrained: If `pretrained="timm"`, load pretrained weights from timm library
            and convert to Tensorflow. Requires timm and torch to be installed. If
            `pretrained` casts to True as bool, load pretrained weights from URL in
            config or the model cache. If `False`, no weights are loaded.
        model_path: Path of model weights to load after model is initialized
        in_channels: Number of input channels for model
        nb_classes: Number of classes for classifier. If set to 0, no classifier is
            used and last layer is pooling layer.
        **kwargs: other kwargs are model specific
    """
    if is_model(model_name):
        cls = model_class(model_name)
        cfg = model_config(model_name)
    else:
        raise RuntimeError("Unknown model (%s)" % model_name)

    if model_path:
        loaded_model = tf.keras.models.load_model(model_path)
    elif pretrained == "timm":
        loaded_model = cls(cfg)
        loaded_model(loaded_model.dummy_inputs)
        load_timm_weights(loaded_model, model_name)
    elif pretrained:
        if cfg.url.endswith(".pth"):
            loaded_model = cls(cfg)
            loaded_model(loaded_model.dummy_inputs)
            load_pth_url_weights(loaded_model, cfg.url)
        elif cfg.url:
            raise NotImplementedError(
                "Download of weights only implemented for PyTorch models."
            )
        else:
            # Load model from cache
            model_path = os.path.join(get_dir(), model_name)
            if not os.path.exists(model_path):
                raise RuntimeError(f"Cannot load model from {model_path}.")
            loaded_model = tf.keras.models.load_model(model_path)
    else:
        loaded_model = None

    # Update config with kwargs
    cfg = deepcopy(cfg)
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logging.warning(
                f"Config for model {model_name} does not have field `{key}`. "
                "Ignoring field."
            )
    if in_channels is not None:
        setattr(cfg, "in_channels", in_channels)
    if nb_classes is not None:
        setattr(cfg, "nb_classes", nb_classes)

    # `keras.Model` kwargs need separate treatment. For now we support only `name`.
    model_kwargs = {}
    if "name" in kwargs:
        model_kwargs["name"] = kwargs["name"]

    # If we have loaded a model and the model has the correct config, then we are done.
    if loaded_model is not None and loaded_model.cfg == cfg:
        return loaded_model

    # Otherwise we build a new model and transfer the weights to it. This is because
    # some parameter changes (in_channels and nb_classes) require changing the shape of
    # some weights or dropping of others. And there might be non-trivial interactions
    # between various parameters, e.g., global_pool can be None only if nb_classes is 0.
    model = cls(cfg, **model_kwargs)
    model(model.dummy_inputs)  # Call model to build layers

    # Now we need to transfer weights from loaded_model to model
    if loaded_model is not None:
        transfer_weights(loaded_model, model)

    return model


def create_preprocessing(model_name: str, dtype: Optional[str] = None) -> Callable:
    """
    Creates a function to preprocess images for a particular model.

    The input to the preprocessing function is assumed to be values in range [0, 255].

    Args:
        model_name: Model for which to create preprocessing function.
        dtype: Output dtype
    """
    if not is_model(model_name):
        raise ValueError(f"Unknown model: {model_name}.")

    cfg = model_config(model_name)
    dtype = dtype or tf.keras.backend.floatx()

    def _preprocess(img: tf.Tensor) -> tf.Tensor:
        img = tf.cast(img, dtype=dtype) / 255.0
        img = (img - cfg.mean) / cfg.std
        return img

    return _preprocess


def transfer_weights(src_model: tf.keras.Model, dst_model: tf.keras.Model):
    """Transfers weights from src_model to dst_model, with special treatment of first
    convolution and classification layers."""
    keep_classifier = src_model.cfg.nb_classes == dst_model.cfg.nb_classes
    dst_classifier = dst_model.cfg.classifier
    if isinstance(dst_classifier, str):  # Some models have multiple classifier heads
        dst_classifier = [dst_classifier]

    # For some weights we may have to apply custom transforms to adapt them to the new
    # model config
    transform_weights = getattr(src_model.cfg, "transform_weights", dict())

    # Some weights should not be transferred as they are created during building
    weights_to_ignore = getattr(dst_model, "keys_to_ignore_on_load_missing", [])

    src_weights = {_strip_prefix(w.name): w.numpy() for w in src_model.weights}
    weight_value_tuples = []
    for dst_weight in dst_model.weights:
        # We store human-friendly names in the model config.
        var_name = _get_layer_name(dst_weight.name)
        w_name = _strip_prefix(dst_weight.name)

        if any(re.search(pat, w_name) is not None for pat in weights_to_ignore):
            # Weights that don't need to be transferred
            pass

        elif var_name in dst_classifier:
            if keep_classifier:
                # We only keep the classifier if the number of classes is the same
                # Otherwise the classifier is not copied over, i.e., we keep the
                # initialization of dst_model.
                weight_value_tuples.append((dst_weight, src_weights[w_name]))

        elif var_name == dst_model.cfg.first_conv:
            # For the first convolution we need to adapt for the number of in_channels.
            if src_model.cfg.in_channels != dst_model.cfg.in_channels:
                raise ValueError("Different number of in_channels not supported yet.")
            weight_value_tuples.append((dst_weight, src_weights[w_name]))

        elif var_name in transform_weights:
            # We check if we need to apply a transform. In that case the weight
            # taken from the model, it is *not* passed to the transform function.
            src_weight = transform_weights[var_name](src_model, dst_model.cfg)
            weight_value_tuples.append((dst_weight, src_weight))

        else:
            # All other weights are simply copied over
            weight_value_tuples.append((dst_weight, src_weights[w_name]))

    # This modifies weights in place
    K.batch_set_value(weight_value_tuples)


def _strip_prefix(name):
    """
    The model name prefix is made unique by TF, i.e., two ResNets will have variables
    'res_net/var:0' and 'res_net_1/var:0'. Here we return 'var:0'.
    """
    return name.split("/", 1)[-1]


def _get_layer_name(name):
    """
    Extracts the name of the layer, which is compared against the config values for
    `first_conv` and `classifier`.

    The input is, e.g., name="res_net/remove/fc/kernel:0". We want to extract "fc".
    """
    name = name.replace(":0", "")  # Remove device IDs
    name = name.replace("/remove/", "/")  # Auxiliary intermediate levels
    name = name.split("/", 1)[-1]  # Remove prefix, e.g., "res_net"
    name = name.rsplit("/", 1)[0]  # Remove last part, e.g., "kernel"
    return name
