"""Data type definitions for TiniTorch."""

from enum import Enum


class DType(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    def __repr__(self):
        return f"tinitorch.{self.value}"


float32 = DType.FLOAT32
float64 = DType.FLOAT64
