import warnings

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


def register_lora_architecture(lora_cls):
    # We assume that the LoRA architecture is a subclass of the original model class.
    model_cls = lora_cls.__base__

    if model_cls in _lora_model_class:
        existing_lora_cls = _lora_model_class[model_cls]
        warnings.warn(
            f"Model class {model_cls} has already registered a LoRA version "
            f"{existing_lora_cls}. Registering {lora_cls} will overwrite this."
        )

    # We assume here that LoRA models have a `cfg_class` class attribute.
    lora_cfg = lora_cls.cfg_class

    _lora_model_class[model_cls] = lora_cls
    _lora_model_base_class[lora_cls] = model_cls
    _lora_model_config[model_cls] = lora_cfg

    return lora_cls


def lora_architecture(model_cls):
    if model_cls not in _lora_model_class:
        raise ValueError(
            f"No LoRA variant has been registered for architecture {model_cls}."
        )
    return _lora_model_class[model_cls]


def lora_base_architecture(lora_cls):
    if lora_cls not in _lora_model_base_class:
        raise ValueError(
            f"The class {lora_cls} is not registered as the LoRA variant of any "
            "architecture."
        )
    return _lora_model_base_class[lora_cls]


def lora_config(model_cls):
    if model_cls not in _lora_model_class:
        raise ValueError(
            f"No LoRA variant has been registered for architecture {model_cls}."
        )
    return _lora_model_config[model_cls]
