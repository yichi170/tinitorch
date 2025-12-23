"""TiniTorch - A tiny neural network library."""

from .creation import arange, empty, ones, randn, zeros
from .device import Device
from .dtype import DType, float32, float64, int32, int64
from .ops import add, div, matmul, mul, neg, relu, sub
from .tensor import Tensor

__version__ = "0.1.0"

__all__ = [
    # Core
    "Tensor",
    "DType",
    "float32",
    "float64",
    "int32",
    "int64",
    "Device",
    # Creation
    "empty",
    "zeros",
    "ones",
    "randn",
    "arange",
    # Ops
    "add",
    "mul",
    "neg",
    "sub",
    "div",
    "matmul",
    "relu",
]
