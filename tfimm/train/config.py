import argparse
import ast
import dataclasses
import logging
import sys
from pathlib import Path

import yaml

from tfimm.train.registry import get_cfg_class


def to_dict_format(cfg):
    """
    A configuration is a nested dictionary with potentially dataclasses as values. This
    function converts dataclasses to dictionaries for easier serialization.
    """
    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
        return to_dict_format(cfg)

    res_cfg = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            res_cfg[key] = to_dict_format(val)
        elif dataclasses.is_dataclass(val):
            res_cfg[key] = dataclasses.asdict(val)
        else:
            res_cfg[key] = val
    return res_cfg


def to_cls_format(cfg):
    """
    Converts a configuration in dictionary format to config classes, using the type
    information provided in "_class" fields.
    """
    res_cfg = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            # First recurse into dictionary before converting to class
            val = to_cls_format(val)

            cls = cfg[f"{key}_class"]
            if cls:  # Now convert to correct class, if class is set
                cls = get_cfg_class(cls)
                res_cfg[key] = cls(**val)
            else:
                res_cfg[key] = None
        else:
            res_cfg[key] = val
    return res_cfg


def to_arg_format(cfg):
    """
    Converts the values of a config in dictionary format to tuples (type, value). This
    is needed to separate dtype from default value for argument parsing.

    We treat `None` and `MISSING` as `str`, which is a useful default for parsing
    command line arguments.

    Example:
    ```
    >>> cfg = {"a": 3, "b": {"c": "s"}}
    >>> print(to_arg_format(cfg))
    {"a": (int, 3), "b": {"c": (str, "s")}}
    ```
    """
    res_cfg = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            res_cfg[key] = to_arg_format(val)
        else:
            tp = type(val) if val not in {None, dataclasses.MISSING} else str
            res_cfg[key] = (tp, val)
    return res_cfg


def check_format(cfg):
    """
    We apply the following checks to configurations in dictionary format:
     - For fields ending in `_class`, map None to empty string.
     - If the `xyz_class` field is present, nesting has to happen and we add the `xyz`
       field and initialize it with an empty dictionary.
     - Nesting is only allowed, if the `_class` field is present, i.e., if `cfg[key]`
       is a dictionary, then `cfg[key + "_class"]` has to exist.
     - If the keys `xyz` and `xyz_class` both exist, then `xyz` has to be a dictionary.
       None is mapped to the empty dictionary. An exception is raised for any other
       values.
    """
    res_cfg = {}
    for key, val in cfg.items():
        if key.endswith("_class"):
            if not isinstance(val, str) and val is not None:
                raise ValueError(f"Value for key {key} should be a string.")
            res_cfg[key] = "" if val is None else val

            stem = key[: -len("_class")]  # Remove suffix
            if stem not in cfg:
                res_cfg[stem] = {}
        elif isinstance(val, dict):
            if key + "_class" not in cfg:
                raise ValueError(f"Nesting only allowed if key `{key}_class` exists.")
            res_cfg[key] = check_format(val)
        else:
            if key + "_class" in cfg:
                if val is not None:
                    raise ValueError(f"Value for key {key} has to be a dict.")
                res_cfg[key] = {}
            else:
                res_cfg[key] = val
    return res_cfg


def add_default_args(cfg):
    """
    When `cfg` is a config in dictionary format with values of the form (type, value),
    this function adds default fields from the configuration dataclasses specified in
    "_class" fields.

    If `cfg = {"data_class": "Foo"}`, the function will look up the config class
    associated to `Foo` and add all fields of the config class as nested dictionary
    `cfg["data"] = {...}`.
    """
    res_cfg = {}
    for key, val in cfg.items():
        if key.endswith("_class"):
            res_cfg[key] = val  # We need to copy the field itself
            if val[1] == "":  # The class has not yet been specified, do nothing.
                continue

            cls = get_cfg_class(val[1])
            stem = key[: -len("_class")]  # Remove suffix
            fields = dataclasses.fields(cls)
            # Fields with missing default value will be assigned value MISSING
            params = {field.name: (field.type, field.default) for field in fields}
            if stem in cfg:
                if not isinstance(cfg[stem], dict):
                    raise ValueError(
                        f"cfg[{stem}] should be a dict, but is {type(cfg[stem])}."
                    )
                # If some fields are already set in the config, we use those values,
                # i.e., whatever is in the config has higher priority over default
                # values. We are calling params.update(), because we don't want to
                # introduce new keys to params.
                for sub_key, sub_val in cfg[stem].items():
                    if sub_key in params:
                        params[sub_key] = sub_val
            # Recursively resolve subclasses
            res_cfg[stem] = add_default_args(params)
        elif isinstance(val, dict) and key + "_class" not in cfg:
            # We need to check for key `xyz` if `xyz_class` is not a key, because
            # otherwise we don't know whether we process key `xyz` first or `xyz_class`.
            res_cfg[key] = add_default_args(val)
        elif key + "_class" not in cfg:
            res_cfg[key] = val
    return res_cfg


def deep_to_flat(cfg):
    """
    Function flattens a nested config in dictionary format by joining keys of nested
    dictionaries with ".".

    For example,
    ```
    >>> cfg = {"a": {"b": 1, "c": 2}, "d": 3}
    >>> print(deep_to_flat(cfg))
    {"a.b": 1, "a.c": 2, "d": 3}
    ```
    """
    res_cfg = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            val = deep_to_flat(val)
            # After recursive flattening, `val` is now a flat dictionary. Add its keys
            # to the current level
            for sub_key, sub_val in val.items():
                res_cfg[f"{key}.{sub_key}"] = sub_val
        else:
            res_cfg[key] = val
    return res_cfg


def flat_to_deep(cfg):
    """
    Function converts a flat config to a nested config in dictionary format. This is
    the inverse of `deep_to_flat()`.
    """
    res_cfg = {}
    # By iterating over the items of `cfg` we resolve one level of nesting.
    for key, val in cfg.items():
        if "." in key:
            root, leaf = tuple(key.split(".", 1))  # Split off the first part
            if root not in res_cfg:
                res_cfg[root] = {}
            res_cfg[root][leaf] = val
        else:
            res_cfg[key] = val

    # Now we iterate again and call the function recursively to resolve deeper levels
    # of nesting.
    for key, val in res_cfg.items():
        if isinstance(val, dict):
            res_cfg[key] = flat_to_deep(val)

    return res_cfg


def dump_config(cfg, filename):
    """Converts a config to nested dictionaries and saves them in yaml format."""
    cfg = to_dict_format(cfg)
    # Create directory for `filename` if it doesn't exist
    Path(filename).parents[0].mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False, sort_keys=False)


def apply_cfg_file(cfg, args):
    """
    We check if a config file is expected (presence of `cfg_file` key in `cfg`). If so,
    see, if an updated config file is passed via command line. Then we read the config
    file and apply the values to the config.
    """
    # First we need to check if the user has supplied a config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default=cfg["cfg_file"])
    namespace, _ = parser.parse_known_args(args)
    cfg_file = namespace.cfg_file

    # No config file is present
    if not cfg_file:
        return cfg

    with open(cfg_file, "r") as f:
        loaded_cfg = yaml.load(f, Loader=yaml.Loader)

    # Updating is easiest done with flattened configs
    cfg = deep_to_flat(cfg)
    loaded_cfg = deep_to_flat(loaded_cfg)
    cfg.update(loaded_cfg)  # Because of deep_to_flat() we are operating on a copy

    # And now back to nested ones
    cfg = flat_to_deep(cfg)
    cfg = check_format(cfg)

    return cfg


def str2bool(v):
    """
    Converts various string to bool, accepting various represenations of True/False.
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def arg_as_tuple(s):
    v = ast.literal_eval(s)
    if type(v) is not tuple:
        raise argparse.ArgumentTypeError(f"Argument {s} is not a tuple")
    return v


def get_arg_parser(cfg):
    """
    Constructs argument parser based on the fields contained in the flattened
    config with values of form (type, value). The values are set as default fields
    for the parser.
    """
    parser = argparse.ArgumentParser(
        description="Auto-initialized argument parser",
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    for arg, (tp, val) in cfg.items():
        kwargs = {"dest": arg, "help": arg}
        if val is not dataclasses.MISSING:
            kwargs["default"] = val
        if tp is bool:
            kwargs["type"] = str2bool
        elif tp is tuple:
            kwargs["type"] = arg_as_tuple
        else:
            kwargs["type"] = tp
        parser.add_argument(f"--{arg}", **kwargs)

    return parser


def parse_args(cfg, *, cfg_class=None, args=None):
    """
    Main function to parse command line arguments. Returns updated config.

    If `cfg_class` is given, we will convert the parsed config to `cfg_class` after
    parsing is finished.

    If `args=None`, we use the passed command line arguments (sys.argv[1:]). If we
    don't want to parse them, we should set `args=[]`. In this case we will read only
    the config file (if present).
    """
    if args is None:
        args = sys.argv[1:]

    # First we convert all dataclasses, etc. to nested dictionaries
    if cfg_class is None:
        cfg_class = type(cfg) if dataclasses.is_dataclass(cfg) else None
    cfg = to_dict_format(cfg)
    cfg = check_format(cfg)

    # Read config file and apply settings.
    if "cfg_file" in cfg:
        cfg = apply_cfg_file(cfg, args)

    nb_unparsed = len(args)
    unparsed = None
    continue_parsing = nb_unparsed > 0
    while continue_parsing:
        # We do the check at the top, so we do one extra round of parsing to add
        # default args to classes that were potentially specified during the last
        # round of parsing.
        continue_parsing = unparsed is None or len(unparsed) > 0

        # We have to convert to argument format in each iteration, because after calling
        # `parser.parse()` the result is a regular config. We lose type information.
        cfg = to_arg_format(cfg)
        # After having parsed some arguments, we may have gained knowledge about
        # which configurations classes are used. Add those parameters (and defaults)
        # to the config now.
        cfg = add_default_args(cfg)
        # Flatten everything in preparation to parsing.
        cfg = deep_to_flat(cfg)

        # Now we can construct a parser and do some parsing
        parser = get_arg_parser(cfg)
        parsed_cfg, unparsed = parser.parse_known_args(args)
        # After this line `parsed_cfg` will be a flat dictionary.
        parsed_cfg = vars(parsed_cfg)

        # All named parameters in argparse are optional. We don't want that, so we
        # check whether all have been supplied.
        for key in cfg.keys():
            if key not in parsed_cfg:
                raise ValueError(f"Argument {key} was not supplied.")
        cfg = parsed_cfg

        if continue_parsing and len(unparsed) >= nb_unparsed:
            raise ValueError(
                "During the last parsing we have not reduced the number of unparsed "
                "arguments. This suggests that either the user has supplied unknown "
                "arguments or that a '_class' argument is missing."
            )
        nb_unparsed = len(unparsed)

        # We convert back to nested dictionaries, because adding default arguments
        # wants to work with nested dictionaries.
        cfg = flat_to_deep(cfg)
        cfg = check_format(cfg)

    cfg = to_cls_format(cfg)
    # If the original config was a dataclass, we need to restore the original type
    if cfg_class:
        cfg = cfg_class(**cfg)
    return cfg


def pprint(cfg, indent=2):
    """Nicely prints nested config."""
    cfg = to_dict_format(cfg)
    for key, val in cfg.items():
        if isinstance(val, dict):
            logging.info(" " * indent + str(key) + ":")
            pprint(val, indent + 2)
        else:
            logging.info(" " * indent + str(key) + "=" + str(val))
