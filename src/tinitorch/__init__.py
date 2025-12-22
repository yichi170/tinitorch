"""TiniTorch - A tiny neural network library."""

from .device import Device
from .dtype import DType, float32, float64, int32, int64
from .ops import add, mul
from .tensor import Tensor

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "DType",
    "float32",
    "float64",
    "int32",
    "int64",
    "Device",
    "add",
    "mul",
]
