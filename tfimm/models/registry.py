"""
Loading models in ``tfimm`` is based on the model registry.
"""
# Copyright 2020 Ross Wightman
# Copyright 2021 Martins Bruveris
import dataclasses
import fnmatch
import re
import sys
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union

__all__ = [
    "compare_available_models_with_timm",
    "is_deprecated",
    "is_model",
    "is_pretrained",
    "list_models",
    "list_modules",
    "model_class",
    "model_config",
    "model_metadata",
    "register_deprecation",
    "register_model",
    "register_model_tag",
    "resolve_model_name",
]

# Dictionaries for model class and configs for the model base names (without tags).
# These are used to register model tags
_model_base_class = {}
_model_base_config = {}
_model_base_module = {}

# Dictionaries giving for each model name the model class and config.
_model_class = {}
_model_config = {}
# Model metadata contains a free-form dictionary of additional information about the
# model, such as licence, inference parameters, etc. This only exists for tagged models.
_model_metadata = {}
# Dictionary mapping model name to name including the default tag.
_model_default_tags = {}
# Dictionary mapping deprecated model names to new name in the model_name.tag format.
_model_deprecations = {}

# Dict of sets to check membership of model in module
_module_to_models = defaultdict(set)
_pretrained_models = set()  # Model names that have pretrained weight url present
_timm_pretrained_models = set()  # Model names that have pretrained weights in timm


def _split_model_name(full_name: str, no_tag: str = "") -> Tuple[str, str]:
    model_name, *tag_list = full_name.split(".", 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def _register_pretrained_weights(cfg):
    """Adds model to the pretrained weight registries."""
    if cfg.url:  # If URL is non-null, we assume it points to pretrained weights
        _pretrained_models.add(cfg.name)
    if cfg.url.startswith("[timm]"):
        _timm_pretrained_models.add(cfg.name)


def register_model(fn=None, *, default_tag: str = ""):
    """
    Decorator for model creation functions. It is used as follows

    .. code-block:: python

        @register_model
        def resnet18():
            cfg = ResNetConfig(name="resnet18.in21k", url="[timm]", ...)
            return ResNet, cfg

    This will register two models: "resnet18" without pretrained weights (url set to "")
    and "resnet18.in21k" with url="[timm]".

    Note that the decorated function must have the same name as defined in the config
    (without tag).

    If the decorator contains the ``default_tag`` parameter, we associate the default
    tag with the model base name (here, "resnet18"). The user needs to call
    ``register_model_tag`` separately to add URL information, etc.

    Args:
        fn: Model creation function.
        default_tag: Default tag to associate with the model.
    """
    # Called with arguments, we return a function that accepts `fn`.
    if fn is None:
        return partial(register_model, default_tag=default_tag)

    # Get model class and model config
    cls, cfg = fn()
    model_name, model_tag = _split_model_name(cfg.name)
    if fn.__name__ != model_name:
        raise ValueError(f"Model name({model_name}) != function name ({fn.__name__}).")
    if model_tag != "" and default_tag != "":
        raise ValueError(
            f"Cannot provide default tag {default_tag}, "
            f"if model name contains tag {model_tag}."
        )

    # Lookup module, where model is defined
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    # Add model function to __all__ in that module
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # Register model without tag to enable later tag registrations.
    _model_base_class[model_name] = cls
    _model_base_config[model_name] = dataclasses.replace(cfg, name=model_name, url="")
    _model_base_module[model_name] = module_name

    # Register model with full name, including tag if present
    _model_class[cfg.name] = cls
    _model_config[cfg.name] = deepcopy(cfg)
    _model_metadata[cfg.name] = {}
    _module_to_models[module_name].add(cfg.name)
    _register_pretrained_weights(cfg)

    if default_tag != "":
        _model_default_tags[model_name] = f"{model_name}.{default_tag}"

    return fn


def register_model_tag(model_name: str, url: str, cfg=None, metadata=None):
    """
    Adds a model tag to the registry. We assume that the model itself has already been
    registered.

    Args:
        model_name: Full model name, including tag, e.g., "resnet18.in21k".
        url: URL with model weights
        cfg: Dictionary with updates to model config, e.g., changes to `nb_classes`.
        metadata: Dictionary with free-form metadata.
    """
    full_name = model_name
    model_name, model_tag = _split_model_name(full_name)
    if model_tag == "":
        raise ValueError(f"Cannot register tag: {full_name} does not contain tag.")

    cfg_updates = cfg or {}
    metadata = metadata or {}

    # Retrieve model config
    cls = _model_base_class[model_name]
    cfg = _model_base_config[model_name]
    module = _model_base_module[model_name]

    # Apply changes to config
    cfg = dataclasses.replace(cfg, name=full_name, url=url, **cfg_updates)

    # Push changes to registry
    _model_class[full_name] = cls
    _model_config[full_name] = cfg
    _model_metadata[full_name] = metadata
    _module_to_models[module].add(full_name)
    _register_pretrained_weights(cfg)


def register_deprecation(old_name: str, new_name: str):
    """Adds a depractation mapping from ``old_name`` to ``new_name``."""
    _model_deprecations[old_name] = new_name


def _natural_key(string_):
    """
    Converts string to list of strings and numbers, i.e.,
        "abc123xyz" -> ["abc", 123, "xyz"]
    to obtain a more natural sort order: "resnet34" comes before "resnet101".
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def list_models(
    name_filter: Union[str, List[str]] = "",
    module: str = "",
    pretrained: Union[bool, str] = False,
    exclude_filters: Union[str, List[str]] = "",
    include_tags: Optional[bool] = None,
):
    """
    Returns list of available model names, sorted alphabetically.

    Args:
        name_filter: Wildcard filter string that works with fnmatch
        module: Limit model selection to a specific sub-module (ie "resnet")
        pretrained: If True only include models with pretrained weights. If "timm",
            only include models with pretrained weights in timm library
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after
            including them with filter
        include_tags: Include pretrained tags in model names (model.tag). If None,
            defaults to True when pretrained=True else False.

    Example:
        model_list("gluon_resnet*") -- returns all models starting with "gluon_resnet"
        model_list("*resnext*", "resnet") -- returns all models with "resnext" in
            "resnet" module
    """
    if include_tags is None:
        # It is unclear if this will stay the default behaviour in the timm code. But
        # it does kind of make sense: Only tagged models are pretrained (the tag can be
        # empty as it is for all non-converted models.
        include_tags = bool(pretrained)

    if pretrained is not False and not include_tags:
        raise ValueError("Cannot list pretrained models without tags.")

    if include_tags:
        if module:
            all_models = _module_to_models[module]
        else:
            all_models = set(_model_class.keys())
    else:
        if module:
            all_models = set(k for k, v in _model_base_module.items() if v == module)
        else:
            all_models = set(_model_base_module.keys())

    if name_filter:
        if not isinstance(name_filter, (tuple, list)):
            name_filter = [name_filter]
        models = set()
        for f in name_filter:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = models.union(include_models)
    else:
        models = all_models

    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = models.difference(exclude_models)

    if pretrained is True:
        models = _pretrained_models.intersection(models)
    elif pretrained == "timm":
        models = _timm_pretrained_models.intersection(models)

    return list(sorted(models, key=_natural_key))


def resolve_model_name(model_name: str) -> str:
    """
    Given a model name, we resolve deprecation mappings and add the default tag if
    present so the returned model name can be looked up in the registry dicts.

    Args:
        model_name: Model name to be resolved.

    Returns:
        Resolved model name.
    """
    # First check for deprecations
    if model_name in _model_deprecations:
        return _model_deprecations[model_name]

    # Then check if a tag is already present
    full_name = model_name
    model_name, model_tag = _split_model_name(full_name)
    if model_tag != "":
        return full_name

    # Without a tag, we first look for a default tag
    if model_name in _model_default_tags:
        return _model_default_tags[model_name]

    # Otherwise return model name without tag.
    return model_name


def is_model(model_name: str) -> bool:
    """
    Check if a model of a given name exists in the registry.
    """
    return resolve_model_name(model_name) in _model_class


def is_deprecated(model_name: str) -> bool:
    """
    Check if a given model name is deprecated in favour of the new model.tag format.
    """
    return model_name in _model_deprecations


def is_pretrained(model_name):
    """Checks if model has pretrained weights."""
    return model_name in _pretrained_models


def model_class(model_name: str):
    """
    Fetch a model class for specified model name.
    """
    return _model_class[resolve_model_name(model_name)]


def model_config(model_name: str):
    """
    Fetch a model config for specified model name.
    """
    return _model_config[resolve_model_name(model_name)]


def model_metadata(model_name: str):
    """
    Fetch the metadata for a specified model name.
    """
    return _model_metadata[resolve_model_name(model_name)]


def list_modules():
    """Return list of module names that contain models."""
    modules = set(_model_base_module.values())
    return list(sorted(modules))


def _to_timm_module_name(module):
    """
    Some modules are called differently in tfimm and timm. This function converts the
    tfimm name to the timm name.
    """
    if module == "vit":
        module = "vision_transformer"
    elif module == "swin":
        module = "swin_transformer"
    return module


def compare_available_models_with_timm(
    name_filter: Union[str, List[str]] = "",
    module: str = "",
    exclude_filters: Union[str, List[str]] = "",
):
    """
    Helper function to list which models have not yet been converted from timm.
    """
    import timm

    tf_models = list_models(
        name_filter=name_filter,
        module=module,
        pretrained="timm",
        exclude_filters=exclude_filters,
        include_tags=True,
    )
    pt_models = timm.list_models(
        filter=name_filter,
        module=_to_timm_module_name(module),
        pretrained=True,
        exclude_filters=exclude_filters,
        include_tags=True,
    )

    tf_models = set(tf_models)
    pt_models = set(pt_models)
    in_timm = set()
    not_in_timm = set()

    for model in tf_models:
        cfg = model_config(model)
        timm_name = cfg.url[len("[timm]") :] or model
        if timm_name in pt_models:
            in_timm.add(timm_name)
            pt_models.remove(timm_name)
        else:
            not_in_timm.add(timm_name)

    print(f"In `timm` and `tfimm`: {len(in_timm)}.")
    print(f"Only `timm`: {len(pt_models)}.")
    print(f"Only `tfimm`: {len(not_in_timm)}.")
    print()
    print(f"Only `timm`:\n{list(sorted(pt_models, key=_natural_key))}\n")
    print(f"Only `tfimm:\n{list(sorted(not_in_timm, key=_natural_key))}\n")
