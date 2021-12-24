"""
This module includes pytorch architectures that are not part of timm, but use the timm
type registry system. We include them here so we can use them in unit tests.
"""
from .pvt import *  # noqa: F401
from .pvt_v2 import *  # noqa: F401
