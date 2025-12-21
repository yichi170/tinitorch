"""Data type definitions for TiniTorch."""

from enum import Enum


class DType(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"

    def __repr__(self):
        return f"tinitorch.{self.value}"


float32 = DType.FLOAT32
float64 = DType.FLOAT64
int32 = DType.INT32
int64 = DType.INT64
