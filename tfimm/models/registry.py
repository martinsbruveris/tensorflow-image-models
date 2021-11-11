""" Model Registry

Based on timm/models/registry.py  by Ross Wightman

Copyright 2020 Ross Wightman
Copyright 2021 Martins Bruveris
"""

import fnmatch
import re
import sys
from collections import defaultdict
from copy import deepcopy
from typing import List, Union

__all__ = [
    "list_models",
    "is_model",
    "is_model_in_modules",
    "list_modules",
    "model_class",
    "model_config",
    "register_model",
]

_model_class = {}
_model_config = {}
# Dict of sets to check membership of model in module
_module_to_models = defaultdict(set)
_model_has_pretrained = set()  # Model names that have pretrained weight url present


def register_model(fn):
    # Get model class and model config
    cls, cfg = fn()
    model_name = cfg.name
    if fn.__name__ != model_name:
        raise ValueError(f"Model name({model_name}) != function name ({fn.__name__}).")

    # Lookup module, where model is defined
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    # Add model function to __all__ in that module
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # Add entries to registry dict/sets
    _model_class[model_name] = cls
    _model_config[model_name] = deepcopy(cfg)
    _module_to_models[module_name].add(model_name)
    if cfg.url:  # If URL is non-null, we assume it points to pretrained weights
        _model_has_pretrained.add(model_name)

    return fn


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(
    name_filter: Union[str, List[str]] = "",
    module: str = "",
    pretrained: bool = False,
    exclude_filters: Union[str, List[str]] = "",
):
    """Returns list of available model names, sorted alphabetically.

    Args:
        name_filter: Wildcard filter string that works with fnmatch
        module: Limit model selection to a specific sub-module (ie 'resnet')
        pretrained: Include only models with pretrained weights if True
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after
            including them with filter

    Example:
        model_list("gluon_resnet*") -- returns all models starting with 'gluon_resnet'
        model_list("*resnext*", "resnet") -- returns all models with 'resnext' in
            'resnet' module
    """
    if module:
        all_models = list(_module_to_models[module])
    else:
        all_models = _model_class.keys()
    if name_filter:
        if not isinstance(name_filter, (tuple, list)):
            name_filter = [name_filter]
        models = set()
        for f in name_filter:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = models.union(include_models)
    else:
        models = set(all_models)
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = models.difference(exclude_models)
    if pretrained:
        models = _model_has_pretrained.intersection(models)
    return list(sorted(models, key=_natural_key))


def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_class


def model_class(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_class[model_name]


def model_config(model_name):
    """Fetch a model config for specified model name
    """
    return _model_config[model_name]


def list_modules():
    """Return list of module names that contain models / model entrypoints
    """
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_model_in_modules(model_name, module_names):
    """Check if a model exists within a subset of modules
    Args:
        model_name (str) - name of model to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained
