import warnings
from functools import partial

# These dictionaries contain the mappings from model class to the corresponding LoRA
# version of the model, i.e., ConvNeXt -> LoRAConvNeXt. The keys are the model class,
# while the values are the class and config class.
#
# Note that this is different from the model registry in `models/registry.py`, where
# keys are specific model names, e.g., "convnext_tiny" and the config dictionary
# contains concrete instances of the configuration classes.
_lora_model_class = dict()
_lora_model_base_class = dict()  # Inverse dict to _lora_model_class
_lora_model_config = dict()


def register_lora_architecture(lora_cls=None, *, base_cls=None):
    """
    Decorator to register a LoRA variant of a model architecture. It is used as follows

    .. code-block:: python

        @register_lora_architecture
        class LoRAResNet(ResNet):
            ...

    This will associate the class ``LoRAResNet`` as the LoRA version of the class
    ``ResNet``. If the LoRA architecture is not created via subclassing, it can be
    specified explicitely.

    .. code-block:: python

        @register_lora_architecture(base_cls=ResNet)
        class LoRAResNet(tf_keras.Model):
            ...

    A model class can be its own LoRA variant, if the model can be created with regular
    or LoRA layers depending on the config. In that case this function needs to be
    invoked after the model has been defined.

    .. code-block:: python

        class FlexibleModel(tf_keras.Model):
            ...

        register_lora_architecture(FlexibleModel, base_cls=FlexibleModel)

    Args:
        lora_cls: LoRA model class. We assume that it has a ``cfg_class`` class
            attribute.
        base_cls: Regular model class. If not provided we use the base class of
            ``lora_cls``.
    """
    # Called with arguments, we return a function that accepts `lora_cls`.
    if lora_cls is None:
        return partial(register_lora_architecture, base_cls=base_cls)

    if base_cls is None:
        # If `base_cls` is not provided, we assume that the LoRA architecture is a
        # subclass of the original model class.
        base_cls = lora_cls.__base__

    if base_cls in _lora_model_class:
        existing_lora_cls = _lora_model_class[base_cls]
        warnings.warn(
            f"Model class {base_cls} has already registered a LoRA version "
            f"{existing_lora_cls}. Registering {lora_cls} will overwrite this."
        )

    # We assume here that LoRA models have a `cfg_class` class attribute.
    lora_cfg = lora_cls.cfg_class

    _lora_model_class[base_cls] = lora_cls
    _lora_model_base_class[lora_cls] = base_cls
    _lora_model_config[base_cls] = lora_cfg

    return lora_cls


def lora_architecture(model_cls):
    """Returns the LoRA model class registered for a given base model class."""
    if model_cls not in _lora_model_class:
        raise ValueError(
            f"No LoRA variant has been registered for architecture {model_cls}."
        )
    return _lora_model_class[model_cls]


def lora_base_architecture(lora_cls):
    """Returns the base class corresponding to the given registered LoRA model class."""
    if lora_cls not in _lora_model_base_class:
        raise ValueError(
            f"The class {lora_cls} is not registered as the LoRA variant of any "
            "architecture."
        )
    return _lora_model_base_class[lora_cls]


def lora_config(model_cls):
    """
    Returns the config class for the LoRA model associated with the given base model
    class.
    """
    if model_cls not in _lora_model_class:
        raise ValueError(
            f"No LoRA variant has been registered for architecture {model_cls}."
        )
    return _lora_model_config[model_cls]
