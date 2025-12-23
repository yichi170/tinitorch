"""Mathematical operations."""

from ..storage import Storage
from ..tensor import Tensor, compute_strides


def _make_result_tensor(storage, shape, dtype, device):
    """Helper to create result tensor from storage."""
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


def add(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_storage = Storage(a.numel(), a.dtype, a.device)

    for i in range(a.numel()):
        result_storage[i] = a._data[i] + b._data[i]

    return _make_result_tensor(result_storage, a.shape, a.dtype, a.device)


def mul(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_storage = Storage(a.numel(), a.dtype, a.device)

    for i in range(a.numel()):
        result_storage[i] = a._data[i] * b._data[i]

    return _make_result_tensor(result_storage, a.shape, a.dtype, a.device)


def neg(a: Tensor) -> Tensor:
    result_storage = Storage(a.numel(), a.dtype, a.device)
    for i in range(a.numel()):
        result_storage[i] = -a._data[i]

    return _make_result_tensor(result_storage, a.shape, a.dtype, a.device)


def sub(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_storage = Storage(a.numel(), a.dtype, a.device)
    for i in range(a.numel()):
        result_storage[i] = a._data[i] - b._data[i]

    return _make_result_tensor(result_storage, a.shape, a.dtype, a.device)


def div(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_storage = Storage(a.numel(), a.dtype, a.device)
    for i in range(a.numel()):
        result_storage[i] = a._data[i] / b._data[i]

    return _make_result_tensor(result_storage, a.shape, a.dtype, a.device)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul only supports 2D tensors")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul shape mismatch: {a.shape} @ {b.shape}, inner dimensions must match"
        )

    m, k = a.shape
    _, n = b.shape
    result_shape = (m, n)

    result_storage = Storage(m * n, a.dtype, a.device)

    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i, p] * b[p, j]
            result_storage[i * n + j] = total

    return _make_result_tensor(result_storage, result_shape, a.dtype, a.device)


def relu(x: Tensor) -> Tensor:
    result_storage = Storage(x.numel(), x.dtype, x.device)
    for i in range(x.numel()):
        val = x._data[i]
        result_storage[i] = val if val > 0 else 0.0

    return _make_result_tensor(result_storage, x.shape, x.dtype, x.device)
