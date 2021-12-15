import argparse
import ast
import sys
from copy import deepcopy
import dataclasses
from dataclasses import asdict, dataclass, is_dataclass

import yaml
from pprint import pprint

MISSING = dataclasses.MISSING
_cfg_classes = {}


def cfg_serializable(cls):
    """
    Registers the configuration datatype associated to a class to a global registry.
    This will be used when loading a configuration from file.
    """
    cls_name = cls.__name__
    _cfg_classes[cls_name] = cls.cfg_class
    return cls


def get_cfg_class(cls):
    """Retrieves the configuration datatype associated to a class."""
    return _cfg_classes[cls]





def dump_config(cfg, filename):
    """Converts a config to nested dictionaries and saves them in yaml format."""
    cfg = convert_cfg_to_dict(cfg)
    with open(filename, "w") as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)


def load_config(filename):
    """
    Loads a config from a yaml file and returns a config dictionary.

    This function will try to recreate the config datatypes using the type registry.
    If it finds a key "xyz_class", it will look up the config class associated to the
    type cfg["xyz_class"] and convert the entry cfg["xyz"] to the corresponding class.

    This means that the "_class" suffix has special meaning and should not be used for
    other purposes.
    """
    with open(filename, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    for key in cfg:
        if key.endswith("_class"):
            stem = key[:-len("_class")]
            cfg_class = get_cfg_class(cfg[key])
            cfg[stem] = cfg_class(**cfg[stem])
    return cfg


# Each class with hyperparameters should have an associated config dataclass with
# parameters. The typehints will be used to create the command line parser (not
# implemented yet).
@dataclass
class TrainerConfig:
    nb_epochs: int
    # Validation
    skip_first_val: bool = False
    validation_every_it: int = -1
    # Checkpointing
    ckpt_every_it: int = -1
    ckpt_to_keep: int = 3
    resume_from_ckpt: bool = True
    # Display
    display_loss_every_it: int = 1000
    verbose: bool = True
    debug_mode: bool = False


# Each class with hyperparameters needs to be decorated with the `@cfg_serializable`
# decorator, which will associate the class and the corresponding config dataclass
# via the `cfg_class` field.
@cfg_serializable
class BasicTrainer:
    cfg_class = TrainerConfig

    # The constructor will in general take only the config object.
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        pass


def cfg_deep_to_flat(cfg):
    """
    Function flattens a nested config in dictionary format by joining keys of nested
    dictionaries with ".".

    For example
    ```
    >>> cfg = {"a": {"b": 1, "c": 2}, "d": 3}
    >>> print(flatten_cfg(cfg))
    {"a.b": 1, "a.c": 2, "d": 3}
    ```
    """
    if not isinstance(cfg, dict):
        return cfg  # Atomic types

    flat_cfg = {}
    for key, val in cfg.items():
        sub_cfg = cfg_deep_to_flat(val)
        if isinstance(sub_cfg, dict):
            for sub_key, sub_val in sub_cfg.items():
                flat_cfg[f"{key}.{sub_key}"] = sub_val
        else:
            flat_cfg[key] = sub_cfg
    return flat_cfg


def cfg_flat_to_deep(cfg):
    """
    Function converts a flat config to a nested config in dictionary format.
    """
    deep_cfg = {}
    # By iterating over the items of `cfg` we resolve one level of nesting.
    for key, val in cfg.items():
        if "." in key:
            root, leaf = tuple(key.split(".", 1))
            if root not in deep_cfg:
                deep_cfg[root] = {}
            deep_cfg[root][leaf] = val
        else:
            deep_cfg[key] = val

    # Now we iterate again and call the function recursively to resolve deeper levels
    # of nesting.
    for key, val in deep_cfg.items():
        if isinstance(val, dict):
            deep_cfg[key] = cfg_flat_to_deep(val)

    return deep_cfg


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f"Argument {s} is not a list")
    return v


def get_arg_parser(cfg):
    """
    Constructs argument parser based on the fields contained in the flattened
    config.
    """
    parser = argparse.ArgumentParser(
        description="Auto-initialized argument parser",
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    for arg, (tp, val) in cfg.items():
        kwargs = {"dest": arg, "help": arg}
        if val is not MISSING:
            kwargs["default"] = val
        if tp is bool:
            kwargs["type"] = str2bool
        elif tp is list:
            kwargs["type"] = str if not val else type(val[0])
            kwargs["nargs"] = "*"
        else:
            kwargs["type"] = tp
        parser.add_argument(f"--{arg}", **kwargs)

    return parser


def cfg_to_arg_format(cfg):
    """
    Converts the values of a config in dictionary format to tuples (type, value).

    ```
    >>> cfg = {"a": 3, "b": {"c": "s"}}
    >>> print(cfg_to_arg_format(cfg))
    {"a": (int, 3), "b": {"c": (str, "s")}}
    ```
    """
    arg_cfg = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            arg_cfg[key] = cfg_to_arg_format(val)
        else:
            tp = type(val) if val not in {None, MISSING} else str
            arg_cfg[key] = (tp, val)
    return arg_cfg


def cfg_add_default_args(cfg):
    """
    When `cfg` is a config in dictionary format with values of the form (type, value),
    this function adds default fields from the configuration dataclasses using "_class"
    fields.
    """
    res_cfg = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            res_cfg[key] = cfg_add_default_args(cfg[key])
        elif key.endswith("_class"):
            res_cfg[key] = val
            if val[1] is MISSING:
                continue

            cls = get_cfg_class(val[1])
            stem = key[:-len("_class")]
            default_cfg = parse_cfg_class_params(cls)
            if stem in cfg:
                if not isinstance(cfg[stem], dict):
                    raise ValueError(
                        f"cfg[{stem}] should be a dict, but is {type(cfg[stem])}."
                    )
                default_cfg.update(cfg[stem])
            res_cfg[stem] = cfg_add_default_args(default_cfg)
        else:
            res_cfg[key] = val
    return res_cfg


def parse_cfg_class_params(cls):
    fields = dataclasses.fields(cls)
    params = {field.name: (field.type, field.default) for field in fields}
    return params


def main():
    cfg = {
        "trainer": TrainerConfig(nb_epochs=10),
        "trainer_class": "BasicTrainer"
    }
    dump_config(cfg, "tmp/cfg.yaml")

    cfg = load_config("tmp/cfg.yaml")
    pprint(cfg)


def main_2():
    # Now we do some command line argument parsing
    cmdline_args = ["--trainer_class=BasicTrainer", "--trainer.nb_epochs=10"]
    cmdline_args = ["--trainer_class=BasicTrainer"]
    # cmdline_args = sys.argv[1:]

    cfg = {
        "trainer_class": MISSING,
    }



    # cfg = cfg_flat_to_deep(cfg)
    # cfg = cfg_to_arg_format(cfg)
    # cfg = cfg_add_default_args(cfg)
    # cfg = cfg_deep_to_flat(cfg)
    # parser = get_arg_parser(cfg)
    # parser.print_help()
    #
    # cfg = vars(parser.parse_args(["--trainer.nb_epochs=10"]))
    # pprint(cfg)


def main_3():
    pprint(parse_cfg_class_params(TrainerConfig))


if __name__ == "__main__":
    main_2()
