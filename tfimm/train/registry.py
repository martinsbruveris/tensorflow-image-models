_classes = {}
_cfg_classes = {}


def cfg_serializable(cls):
    """
    Registers the configuration datatype associated to a class to a global registry.
    This will be used when loading a configuration from file.
    """
    cls_name = cls.__name__
    _classes[cls_name] = cls
    _cfg_classes[cls_name] = cls.cfg_class
    return cls


def get_cfg_class(cls):
    """Retrieves the configuration datatype associated to a class."""
    return _cfg_classes[cls]


def get_class(cls):
    """Retrieves the registered class."""
    return _classes[cls]
