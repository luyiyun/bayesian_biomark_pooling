from importlib.util import find_spec
from .bbp import BBP
from .embp import EMBP


FLAG_TORCH = find_spec("torch") is not None
__all__ = ["BBP", "EMBP"]
