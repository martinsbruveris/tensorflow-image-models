from copy import deepcopy
from typing import Optional, Union

import tensorflow as tf
from tensorflow.python.keras import backend as K

from tfimm.models.registry import is_model, model_class, model_config
from tfimm.utils import load_timm_weights


# TODO: Implement in_chans, to work with both timm as well as saved models
def create_model(
    model_name: str,
    pretrained: Union[bool, str] = False,
    model_path: str = "",
    *,
    in_chans: Optional[int] = None,
    nb_classes: Optional[int] = None,
    **kwargs,
):
    """Creates a model.

    Args:
        model_name: Name of model to instantiate
        pretrained: If True, load pretrained weights from URL in config. If "timm",
            load pretrained weights from timm library and convert to Tensorflow.
            Requires timm and torch to be installed. If False, no weights are loaded.
        model_path: Path of model weights to load after model is initialized
        in_chans: Number of input channels for model
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
    elif pretrained is True:
        raise NotImplementedError(
            "Automatic loading of pretrained weights only implemented from timm."
        )
    elif pretrained == "timm":
        loaded_model = cls(cfg)
        loaded_model(loaded_model.dummy_inputs)
        loaded_model = load_timm_weights(loaded_model, model_name)
    else:
        loaded_model = None

    # Update config with kwargs
    cfg = deepcopy(cfg)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    if in_chans is not None:
        setattr(cfg, "in_chans", in_chans)
    if nb_classes is not None:
        setattr(cfg, "nb_classes", nb_classes)

    # If we have loaded a model and the model has the correct config, then we are done.
    if loaded_model is not None and loaded_model.cfg == cfg:
        return loaded_model

    # Otherwise we build a new model and transfer the weights to it. This is because
    # some parameter changes (in_chans and nb_classes) require changing the shape of
    # some weights or dropping of others. And there might be non-trivial interactions
    # between various parameters, e.g., global_pool can be None only if nb_classes is 0.
    model = cls(cfg)
    model(model.dummy_inputs)  # Call model to build layers

    # Now we need to transfer weights from loaded_model to model
    if loaded_model is not None:
        transfer_weigths(loaded_model, model)

    return model


def transfer_weigths(src_model: tf.keras.Model, dst_model: tf.keras.Model):
    """Transfers weights from src_model to dst_model, with special treatment of first
    convolution and classification layers."""
    keep_classifier = src_model.cfg.nb_classes == dst_model.cfg.nb_classes
    dst_classifier = dst_model.cfg.classifier
    if isinstance(dst_classifier, str):  # Some models have multiple classifier heads
        dst_classifier = [dst_classifier]

    src_weights = {_strip_prefix(w.name): w.numpy() for w in src_model.weights}
    weight_value_tuples = []
    for dst_weight in dst_model.weights:
        # We store human-friendly names in the model config.
        var_name = _get_layer_name(dst_weight.name)
        w_name = _strip_prefix(dst_weight.name)

        if var_name in dst_classifier:
            if keep_classifier:
                # We only keep the classifier if the number of classes is the same
                # Otherwise the classifier is not copied over, i.e., we keep the
                # initialization of dst_model.
                weight_value_tuples.append((dst_weight, src_weights[w_name]))

        elif var_name == dst_model.cfg.first_conv:
            # For the first convolution we need to adapt for the number of in_chans.
            if src_model.cfg.in_chans != dst_model.cfg.in_chans:
                raise ValueError("Different number of in_chans not supported yet.")
            weight_value_tuples.append((dst_weight, src_weights[w_name]))

        else:
            # All other weights are copied over
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
