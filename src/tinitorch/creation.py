"""Tensor creation functions."""

import random

from .device import Device
from .dtype import DType
from .storage import Storage
from .tensor import Tensor, compute_strides


def _make_tensor(storage, shape, dtype, device):
    result = Tensor.__new__(Tensor)
    result._data = storage
    result._shape = shape
    result._strides = compute_strides(shape)
    result._offset = 0
    result.dtype = dtype
    result.device = device
    result.requires_grad = False
    result.grad = None
    result.grad_fn = None
    return result


def empty(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device(device)

    numel = 1
    for dim in shape:
        numel *= dim

    storage = Storage(numel, dtype, device)
    return _make_tensor(storage, shape, dtype, device)


def zeros(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device(device)

    numel = 1
    for dim in shape:
        numel *= dim

    storage = Storage(numel, dtype, device)
    storage.fill(0.0)
    return _make_tensor(storage, shape, dtype, device)


def ones(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device(device)

    numel = 1
    for dim in shape:
        numel *= dim

    storage = Storage(numel, dtype, device)
    storage.fill(1.0)
    return _make_tensor(storage, shape, dtype, device)


def randn(
    *shape: int,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
) -> Tensor:
    if isinstance(device, str):
        device = Device(device)

    numel = 1
    for dim in shape:
        numel *= dim

    storage = Storage(numel, dtype, device)

    for i in range(numel):
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        import math

        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        storage[i] = z

    return _make_tensor(storage, shape, dtype, device)


def arange(
    start: float,
    end: float | None = None,
    step: float = 1.0,
    dtype: DType = DType.FLOAT32,
    device: str | Device = "cpu",
) -> Tensor:
    """Create 1D tensor with values from start to end."""
    if end is None:
        end = start
        start = 0.0

    if isinstance(device, str):
        device = Device(device)

    numel = int((end - start) / step)
    if numel <= 0:
        numel = 0

    storage = Storage(numel, dtype, device)
    current = start
    for i in range(numel):
        storage[i] = current
        current += step

    return _make_tensor(storage, (numel,), dtype, device)
