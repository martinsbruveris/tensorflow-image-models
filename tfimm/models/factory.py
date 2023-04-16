import logging
import re
from copy import deepcopy
from typing import Callable, Optional

import tensorflow as tf
from tensorflow.python.keras import backend as K

from tfimm.models.registry import is_model, model_class, model_config
from tfimm.utils import cached_model_path, load_pth_url_weights, load_timm_weights


def create_model(
    model_name: str,
    pretrained: bool = False,
    model_path: str = "",
    *,
    in_channels: Optional[int] = None,
    nb_classes: Optional[int] = None,
    **kwargs,
) -> tf.keras.Model:
    """
    Creates a model.

    Args:
        model_name: Name of model to instantiate
        pretrained: If ``True``, load pretrained weights as specified by the ``url``
            field in config. We will check the cache first and download weights only
            if they cannot be found in the cache.

            If ``url`` is ``[timm]``, the weights will be downloaded from ``timm`` and
            converted to TensorFlow. Requires ``timm`` and ``torch`` to be installed.
            If ``url`` starts with ``[pytorch]``, the weights are in PyTorch format
            and ``torch`` needs to be installed to convert them.
        model_path: Path of model weights to load after model is initialized. This takes
            over ``pretrained``.
        in_channels: Number of input channels for model. If ``None``, use default
            provided by model.
        nb_classes: Number of classes for classifier. If set to 0, no classifier is
            used and last layer is pooling layer. If ``None``, use default provided by
            model.
        **kwargs: Other kwargs are model specific.

    Returns:
        The created model.
    """
    if not is_model(model_name):
        raise RuntimeError(f"Unknown model {model_name}.")

    cls = model_class(model_name)
    cfg = model_config(model_name)

    if model_path:
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
    elif pretrained:
        # First try loading model from cache
        model_path = cached_model_path(model_name)
        if model_path:
            loaded_model = tf.keras.models.load_model(model_path)
        elif cfg.url.startswith("[timm]"):
            loaded_model = cls(cfg)
            loaded_model(loaded_model.dummy_inputs)
            # Url can be "[timm]timm_model_name" or "[timm]" in which case we default
            # to model_name.
            timm_model_name = cfg.url[len("[timm]") :] or model_name
            load_timm_weights(loaded_model, timm_model_name)
        elif cfg.url.startswith("[pytorch]"):
            url = cfg.url[len("[pytorch]") :]
            loaded_model = cls(cfg)
            loaded_model(loaded_model.dummy_inputs)
            load_pth_url_weights(loaded_model, url)
        else:
            raise NotImplementedError(
                "Model not found in cache. Download of weights only implemented for "
                "PyTorch models."
            )
    else:
        loaded_model = None

    # Update config with kwargs
    cfg = deepcopy(cfg)
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logging.warning(
                f"Config for {model_name} does not have field `{key}`. Ignoring field."
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

    # Otherwise, we build a new model and transfer the weights to it. This is because
    # some parameter changes (in_channels and nb_classes) require changing the shape of
    # some weights or dropping of others. And there might be non-trivial interactions
    # between various parameters, e.g., global_pool can be None only if nb_classes is 0.
    model = cls(cfg, **model_kwargs)
    model(model.dummy_inputs)  # Call model to build layers

    # Now we need to transfer weights from loaded_model to model
    if loaded_model is not None:
        transfer_weights(loaded_model, model)

    return model


def create_preprocessing(
    model_name: str,
    *,
    in_channels: Optional[float] = None,
    dtype: Optional[str] = None,
) -> Callable:
    """
    Creates a function to preprocess images for a particular model.

    The input to the preprocessing function is assumed to be values in range [0, 255].

    Args:
        model_name: Model for which to create preprocessing function.
        in_channels: Number of input channels to model
        dtype: Output dtype.

    Returns:
        Callable that operates on single images as well as batches.
    """
    if not is_model(model_name):
        raise ValueError(f"Unknown model: {model_name}.")

    cfg = model_config(model_name)
    dtype = dtype or tf.keras.backend.floatx()

    def _adapt_vector(v, n):
        """Adapts vector v to length n by repeating as necessary."""
        v = tf.convert_to_tensor(v, dtype=dtype)
        m = tf.shape(v)[0]
        nb_repeats = n // m + 1
        v = tf.tile(v, [nb_repeats])
        v = v[:n]
        return v

    in_channels = in_channels or cfg.in_channels
    mean = _adapt_vector(cfg.mean, in_channels)
    std = _adapt_vector(cfg.std, in_channels)

    def _preprocess(img: tf.Tensor) -> tf.Tensor:
        img = tf.cast(img, dtype=dtype) / 255.0
        img = (img - mean) / std
        return img

    return _preprocess


def transfer_weights(src_model: tf.keras.Model, dst_model: tf.keras.Model):
    """
    Transfers weights from ``src_model`` to ``dst_model``, with special treatment of
    first convolution and classification layers. The ``dst_model`` is modified in place.

    Args:
        src_model: Source model.
        dst_model: Destination model, modified in place
    """
    dst_first_conv = dst_model.cfg.first_conv

    if hasattr(src_model.cfg, "nb_classes") and hasattr(dst_model.cfg, "nb_classes"):
        keep_classifier = src_model.cfg.nb_classes == dst_model.cfg.nb_classes
    else:
        # It doesn't really matter what we set this to, since if a model doesn't have
        # a nb_classes parameter, it most likely won't have a classifier parameter
        # set either
        keep_classifier = True
    dst_classifier = getattr(dst_model.cfg, "classifier", [])
    if isinstance(dst_classifier, str):  # Some models have multiple classifier heads
        dst_classifier = [dst_classifier]

    # For some weights we may have to apply custom transforms to adapt them to the new
    # model config
    transform_weights = getattr(src_model.cfg, "transform_weights", dict())

    # Some weights should not be transferred as they are created during building
    weights_to_ignore = getattr(dst_model, "keys_to_ignore_on_load_missing", [])

    src_weights = {_get_weight_name(w.name): w.numpy() for w in src_model.weights}
    weight_value_tuples = []
    for dst_weight in dst_model.weights:
        # We store human-friendly names in the model config.
        layer_name = _get_layer_name(dst_weight.name)
        w_name = _get_weight_name(dst_weight.name)

        if any(re.search(pat, w_name) is not None for pat in weights_to_ignore):
            # Weights that don't need to be transferred
            pass

        elif layer_name in dst_classifier:
            if keep_classifier:
                # We only keep the classifier if the number of classes is the same
                # Otherwise the classifier is not copied over, i.e., we keep the
                # initialization of dst_model.
                weight_value_tuples.append((dst_weight, src_weights[w_name]))

        elif layer_name == dst_first_conv:
            src_weight = _transform_first_conv(
                src_weights[w_name], dst_model.cfg.in_channels
            )
            weight_value_tuples.append((dst_weight, src_weight))

        elif w_name in transform_weights:
            # We check if we need to apply a transform. In that case we call the
            # transform function passing the source model, source weight and target
            # config.
            src_weight = transform_weights[w_name](
                src_model, src_weights[w_name], dst_model.cfg
            )
            weight_value_tuples.append((dst_weight, src_weight))

        else:
            # All other weights are simply copied over
            weight_value_tuples.append((dst_weight, src_weights[w_name]))

    # This modifies weights in place
    K.batch_set_value(weight_value_tuples)


def _get_layer_name(name):
    """
    Extracts the name of the layer, which is compared against the config values for
    `first_conv` and `classifier`.

    The input is, e.g., name="res_net/remove/fc/kernel:0". We want to extract "fc".
    """
    name = name.replace(":0", "")  # Remove device IDs
    name = name.replace("/remove/", "/")  # Auxiliary intermediate levels
    # The model name prefix is made unique by TF, i.e., two ResNets will have variables
    # 'res_net/var:0' and 'res_net_1/var:0'. Here we remove the prefix.
    name = name.split("/", 1)[-1]  # Remove prefix, e.g., "res_net"
    name = name.rsplit("/", 1)[0]  # Remove last part, e.g., "kernel"
    return name


def _get_weight_name(name):
    """
    Extracts the name of the weight, which is compared against the config values for
    `transfer_weights`. The difference with `_get_weights_name` is that we preserve
    the last part.

    The input is, e.g., name="res_net/fc/kernel:0". We want to extract "fc/kernel".
    """
    name = name.replace(":0", "")  # Remove device IDs
    name = name.split("/", 1)[-1]  # Remove prefix, e.g., "res_net"
    return name


def _transform_first_conv(weight, in_channels):
    """
    Adapts `weight` to have `in_channels` either by truncating or by repeating
    across channel dimensions.
    """
    if tf.rank(weight) != 4:
        # We don't need to adapt biases, because they don't depend on input channels
        return weight

    # Convolutional kernels have shape (H, W, in_channels, out_channels)
    src_channels = tf.shape(weight)[2]
    if in_channels == src_channels:
        # Nothing to adapt here...
        pass
    elif in_channels == 1:
        # For single-channel input, we sum the existing channels. We don't average in
        # order to preserve weight statistics
        weight = tf.reduce_sum(weight, axis=2, keepdims=True)
    else:
        nb_repeats = in_channels // src_channels + 1
        weight = tf.tile(weight, [1, 1, nb_repeats, 1])
        weight = weight[:, :, :in_channels, :]
        weight *= tf.cast(src_channels / in_channels, tf.float32)
    return weight
