_classes = {}
_cfg_classes = {}


def cfg_serializable(cls):
    """
    Registers the configuration datatype. The configuration class can either be
    associated to a parent class (whose configuration it is) or be standalone.

    In the first case we decorate a class with a `cfg_class` attribute
    ```
    @dataclass
    class DatasetConfig:
        path: str

    @cfg_serializable
    class Dataset:
        cfg_class = DatasetConfig

        def __init__(self, cfg: DatasetConfig):
            ...
    ```

    In the second case, we have only a config class (see `Timekeeping`), decorated
    directly.
    ```
    @cfg_serializable
    @dataclass
    class Timekeeping
        nb_epochs: int
    ```

    We will use the information to parse configurations from file or from command line
    arguments.
    """
    cls_name = cls.__name__

    if hasattr(cls, "cfg_class"):
        # Here we are dealing with a class, that has an associated config class.
        # First we associate the class name with the class and the config type
        _classes[cls_name] = cls
        _cfg_classes[cls_name] = cls.cfg_class

        # We also register the config class itself
        cfg_class_name = cls.cfg_class.__name__
        _cfg_classes[cfg_class_name] = cls.cfg_class
    else:
        # Here we are dealing with a standalone configuration class
        _cfg_classes[cls_name] = cls
    return cls


def get_cfg_class(cls):
    """Retrieves the configuration datatype associated to a class."""
    return _cfg_classes[cls]


def get_class(cls):
    """Retrieves the registered class."""
    return _classes[cls]
