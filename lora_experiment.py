import logging
from typing import Optional
import tensorflow as tf
from copy import deepcopy

from tfimm.layers.lora import lora_convnext_tiny

from tfimm.models.registry import is_model, model_class, model_config
from tfimm.utils import cached_model_path, load_pth_url_weights, load_timm_weights
from tfimm.models.factory import transfer_weights


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

    # cls = model_class(model_name)
    #cls = LoRAConvNeXt
    #cls = convnext.ConvNeXt
    #cfg = model_config(model_name)

    cls, cfg = lora_convnext_tiny()

    if model_path:
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
    elif pretrained:
        # First try loading model from cache
        model_path = cached_model_path(model_name)
        print(model_path)
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

def main():
    #cls, cfg = convnext.convnext_tiny()

    #model = LoRAConvNeXt(cfg=cfg)
    #model(model.dummy_inputs)

    model = create_model("convnext_tiny", pretrained="timm")
    model(model.dummy_inputs)

    merged_model = model.merge_weights()
    merged_model(merged_model.dummy_inputs)

if __name__ == "__main__":
    main()