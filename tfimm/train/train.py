from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass

import yaml

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


def convert_cfg_to_dict(cfg):
    """
    A configuration is a nested dictionary with potentially dataclasses as values. This
    function converts dataclasses to dictionaries for easier serialization.

    We only convert dataclasses at the root of the config dictionary. This can be
    extended to a more recursive lookup if needed in the future.
    """
    cfg = deepcopy(cfg)
    for key in cfg:
        if is_dataclass(cfg[key]):
            cfg[key] = asdict(cfg[key])
    return cfg


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


def main():
    cfg = {
        "trainer": TrainerConfig(nb_epochs=10),
        "trainer_class": "BasicTrainer"
    }
    dump_config(cfg, "tmp/cfg.yaml")

    cfg = load_config("tmp/cfg.yaml")
    print(cfg)


if __name__ == "__main__":
    main()
