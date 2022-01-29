import os
from typing import Optional

ENV_TFIMM_HOME = "TFIMM_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_tfimm_dir = None


def get_dir() -> str:
    """
    Get the ``tfimm`` cache directory used for storing downloaded models & weights.

    If ``tfimm.set_dir`` has not been called, default path is ``$TFIMM_HOME``, where
    environment variable ``$TFIMM_HOME`` defaults to ``$XDG_CACHE_HOME/tfimm``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specifications of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment variable is
    not set.
    """
    if _tfimm_dir is not None:
        return _tfimm_dir

    cache_home = os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR)
    tfimm_home = os.path.join(cache_home, "tfimm")
    tfimm_home = os.getenv(ENV_TFIMM_HOME, tfimm_home)
    tfimm_home = os.path.expanduser(tfimm_home)
    return tfimm_home


def set_dir(d: str):
    """
    Optionally set the ``tfimm`` models directory used to save downloaded models &
    weights.

    Args:
        d: Path to a local folder to save downloaded models & weights.
    """
    global _tfimm_dir
    _tfimm_dir = d


def model_path_cache(model_name: str) -> Optional[str]:
    """
    Checks if the weights for model ``model_name`` are cached. If so, we return the
    path to the weights, otherwise return ``None``.

    Args:
        model_name: Model to be queried.
    """
    model_path = os.path.join(get_dir(), model_name)
    if os.path.exists(model_path):
        return model_path
    else:
        return None
