"""Tensor creation functions."""

import random

from .device import Device
from .dtype import DType
from .tensor import Tensor


def empty(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
    backend: str | None = None,
) -> Tensor:
    if isinstance(device, str):
        device = Device(device)

    numel = 1
    for dim in shape:
        numel *= dim

    data = _build_nested_list(shape, 0.0)
    return Tensor(data, dtype=dtype, device=device, backend=backend)


def zeros(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
    backend: str | None = None,
) -> Tensor:
    data = _build_nested_list(shape, 0.0)
    return Tensor(data, dtype=dtype, device=device, backend=backend)


def ones(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
    backend: str | None = None,
) -> Tensor:
    data = _build_nested_list(shape, 1.0)
    return Tensor(data, dtype=dtype, device=device, backend=backend)


def randn(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
    backend: str | None = None,
) -> Tensor:
    import math

    def random_normal():
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    data = _build_nested_list(shape, random_normal)
    return Tensor(data, dtype=dtype, device=device, backend=backend)


def arange(
    start: float,
    end: float | None = None,
    step: float = 1.0,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
    backend: str | None = None,
) -> Tensor:
    """Create 1D tensor with values from start to end."""
    if end is None:
        end = start
        start = 0.0

    numel = int((end - start) / step)
    if numel <= 0:
        numel = 0

    data = [start + i * step for i in range(numel)]
    return Tensor(data, dtype=dtype, device=device, backend=backend)


def _build_nested_list(shape: tuple, fill_value) -> list:
    if len(shape) == 0:
        return fill_value() if callable(fill_value) else fill_value

    if len(shape) == 1:
        if callable(fill_value):
            return [fill_value() for _ in range(shape[0])]
        return [fill_value] * shape[0]

    return [_build_nested_list(shape[1:], fill_value) for _ in range(shape[0])]
