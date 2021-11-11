from copy import deepcopy
from typing import Optional, Union

from tfimm.models.registry import is_model, model_class, model_config
from tfimm.utils import load_timm_weights


# TODO: Implement in_chans, to work with both timm as well as models in h5 format.
# TODO: Implement nb_classes, to work with both timm as well as models in h5 format.
def create_model(
    model_name: str,
    pretrained: Union[bool, str] = False,
    model_path: str = "",
    *,
    in_chans: Optional[int] = None,
    nb_classes: Optional[int] = None,
    global_pool: Optional[str] = None,
    **kwargs,
):
    """Creates a model

    Args:
        model_name: Name of model to instantiate
        pretrained: If True, load pretrained weights from URL in config. If "timm",
            load pretrained weights from timm library and convert to Tensorflow.
            Requires timm and torch to be installed. If False, no weights are loaded.
        model_path: Path of model weights to load after model is initialized
        in_chans: Number of input channels for model
        nb_classes: Number of classes for classifier. If set to 0, no classifier is
            used and last layer is pooling layer.
        global_pool: Global pooling type ("avg" or "max"). If set to None (or ""), no
            pooling layer is used.
        **kwargs: other kwargs are model specific
    """
    if is_model(model_name):
        cls = model_class(model_name)
        cfg = model_config(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    # Update config with kwargs
    # TODO: We need to be very careful here...
    cfg = deepcopy(cfg)
    for key, value in kwargs.items():
        setattr(cfg, key, value)

    # Create model
    model = cls(cfg)

    if pretrained is True:
        raise NotImplementedError(
            "Automatic loading of pretrained weights only implemented from timm."
        )
    elif pretrained == "timm":
        model = load_timm_weights(model, model_name)

    if model_path:
        model.load_weights(model_path)

    return model
