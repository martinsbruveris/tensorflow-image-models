import os

ENV_TFIMM_HOME = "TFIMM_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_tfimm_dir = None


def get_dir() -> str:
    """
    Get the TFIMM cache directory used for storing downloaded models & weights.

    If `tfimm.set_dir` is not called, default path is `$TFIMM_HOME`, where environment
    variable `$TFIMM_HOME` defaults to `$XDG_CACHE_HOME/tfimm`. `$XDG_CACHE_HOME`
    follows the X Design Group specifications of the Linux filesystem layout, with a
    default value `~/.cache` if the environment variable is not set.
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
    Optionally set the TFIMM models directory used to save downloaded models & weights.

    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    global _tfimm_dir
    _tfimm_dir = d
