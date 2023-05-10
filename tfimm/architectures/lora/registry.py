import warnings

# These dictionaries contain the mappings from model class to the corresponding LoRA
# version of the model, i.e., "ConvNeXt" -> LoRAConvNeXt. The keys are the names of the
# model class, while the values are the class and config class.
#
# Note that this is different from the model registry in `models/registry.py`, where
# keys are specific model names, e.g., "convnext_tiny" and the config dictionary
# contains concrete instances of the configuration classes.
_lora_model_class = dict()
_lora_model_config = dict()


def register_lora_architecture(lora_cls):
    # We assume that the LoRA architecture is a subclass of the original model class.
    model_cls = lora_cls.__base__

    model_name = model_cls.__name__
    if model_name in _lora_model_class:
        existing_lora_cls = _lora_model_class[model_name]
        warnings.warn(
            f"Model class {model_name} has already registered a LoRA version "
            f"{existing_lora_cls.__name__}. Registering {lora_cls.__name__} will "
            "overwrite this."
        )

    # We assume here that LoRA models have a `cfg_class` class attribute.
    lora_cfg = lora_cls.cfg_class

    _lora_model_class[model_name] = lora_cls
    _lora_model_config[model_name] = lora_cfg

    return lora_cls


def lora_architecture(model_cls):
    model_name = model_cls.__name__
    if model_name not in _lora_model_class:
        raise ValueError(
            f"No LoRA variant has been registered for architecture {model_name}."
        )
    lora_cls = _lora_model_class[model_name]
    return lora_cls


def lora_config(model_cls):
    model_name = model_cls.__name__
    if model_name not in _lora_model_class:
        raise ValueError(
            f"No LoRA variant has been registered for architecture {model_name}."
        )
    lora_cfg = _lora_model_config[model_name]
    return lora_cfg
