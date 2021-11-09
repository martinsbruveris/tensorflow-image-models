from copy import deepcopy
from typing import Union

from tfimm.models.registry import is_model, model_class, model_config
from tfimm.utils import load_timm_weights


def create_model(
    model_name: str,
    pretrained: Union[bool, str] = False,
    model_path: str = "",
    **kwargs,
):
    """Creates a model

    Args:
        model_name: Name of model to instantiate
        pretrained: If True, load pretrained weights from URL in config. If "timm",
            load pretrained weights from timm library and convert to Tensorflow.
            Requires timm and torch to be installed. If False, no weights are loaded.
        model_path: Path of model weights to load after model is initialized

    Keyword Args:
        in_chans (int): Number of input channels for model
        drop_rate (float): Dropout rate for training
        global_pool (str): Global pool type
        **kwargs: other kwargs are model specific
    """
    if is_model(model_name):
        cls = model_class(model_name)
        cfg = model_config(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    # Update config with kwargs
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
