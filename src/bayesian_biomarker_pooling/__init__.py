from importlib.util import find_spec
from .bbp import BBP


FLAG_TORCH = find_spec("torch") is not None
__all__ = ["BBP"]
