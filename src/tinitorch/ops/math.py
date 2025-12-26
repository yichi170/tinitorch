"""Mathematical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor


def _result_tensor(data: list, source: Tensor, shape: tuple = None) -> Tensor:
    from ..tensor import Tensor

    if shape is None:
        shape = source.shape
    nested = _build_nested(data, shape)
    return Tensor(
        nested,
        dtype=source._dtype,
        device=source._device,
        backend=source._backend,
    )


def _build_nested(flat_data: list, shape: tuple) -> list:
    if len(shape) == 0:
        return flat_data[0] if flat_data else 0.0
    if len(shape) == 1:
        return flat_data[: shape[0]]

    sub_size = 1
    for dim in shape[1:]:
        sub_size *= dim

    result = []
    for i in range(shape[0]):
        start = i * sub_size
        end = start + sub_size
        result.append(_build_nested(flat_data[start:end], shape[1:]))
    return result


def add(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_data = [a_val + b_val for a_val, b_val in zip(a.flat_iter(), b.flat_iter())]
    return _result_tensor(result_data, a)


def mul(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_data = [a_val * b_val for a_val, b_val in zip(a.flat_iter(), b.flat_iter())]
    return _result_tensor(result_data, a)


def neg(a: Tensor) -> Tensor:
    result_data = [-val for val in a.flat_iter()]
    return _result_tensor(result_data, a)


def sub(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_data = [a_val - b_val for a_val, b_val in zip(a.flat_iter(), b.flat_iter())]
    return _result_tensor(result_data, a)


def div(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_data = [a_val / b_val for a_val, b_val in zip(a.flat_iter(), b.flat_iter())]
    return _result_tensor(result_data, a)


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

    result_data = []
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i, p] * b[p, j]
            result_data.append(total)

    return _result_tensor(result_data, a, result_shape)


def relu(x: Tensor) -> Tensor:
    result_data = [val if val > 0 else 0.0 for val in x.flat_iter()]
    return _result_tensor(result_data, x)
