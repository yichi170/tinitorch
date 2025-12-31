"""Mathematical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor


def _dispatch_cpp(op: str, *tensors: Tensor) -> Tensor:
    from .. import _C

    tensor = tensors[0]
    result_impl = _C.dispatch(
        op,
        _C.Device.CPU,  # TODO: get from tensor
        _C.DType.Float32,  # TODO: get from tensor dtype
        [t._impl for t in tensors],
    )
    return tensor._wrap_impl(result_impl)


def _use_cpp_dispatch(*tensors: Tensor) -> bool:
    return all(t._backend == "cpp" for t in tensors)


# ============================================================
# Broadcasting utilities
# ============================================================


def broadcast_shapes(shape_a: tuple, shape_b: tuple) -> tuple:
    """Compute broadcasted output shape."""
    ndim = max(len(shape_a), len(shape_b))
    # Pad with 1s on the left
    shape_a = (1,) * (ndim - len(shape_a)) + shape_a
    shape_b = (1,) * (ndim - len(shape_b)) + shape_b

    result = []
    for a, b in zip(shape_a, shape_b):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape_a} and {shape_b}")
    return tuple(result)


def _broadcast_strides(shape: tuple, strides: tuple, target_shape: tuple) -> tuple:
    """Compute strides for broadcasting. Stride=0 means repeat that dimension."""
    ndim = len(target_shape)
    offset = ndim - len(shape)
    result = [0] * ndim

    for i, s in enumerate(shape):
        target_idx = offset + i
        if s == target_shape[target_idx]:
            result[target_idx] = strides[i]
        elif s == 1:
            result[target_idx] = 0  # Broadcast: repeat this element
        else:
            raise ValueError("Cannot broadcast")
    return tuple(result)


def _flat_to_indices(flat_idx: int, shape: tuple) -> tuple:
    """Convert flat index to multi-dim indices."""
    indices = []
    for dim in reversed(shape):
        indices.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(indices))


def _compute_broadcast_index(indices: tuple, strides: tuple) -> int:
    """Compute linear index using strides (stride=0 means broadcast)."""
    return sum(i * s for i, s in zip(indices, strides))


def _broadcast_binary_op(a: Tensor, b: Tensor, op) -> Tensor:
    """Apply binary op with broadcasting (Python backend)."""

    out_shape = broadcast_shapes(a.shape, b.shape)

    a_strides = _broadcast_strides(a.shape, a.strides, out_shape)
    b_strides = _broadcast_strides(b.shape, b.strides, out_shape)

    numel = 1
    for d in out_shape:
        numel *= d

    result_data = []
    for i in range(numel):
        indices = _flat_to_indices(i, out_shape)
        a_idx = _compute_broadcast_index(indices, a_strides)
        b_idx = _compute_broadcast_index(indices, b_strides)

        a_val = a._impl.get_flat(a._impl.offset + a_idx)
        b_val = b._impl.get_flat(b._impl.offset + b_idx)
        result_data.append(op(a_val, b_val))

    return _result_tensor(result_data, a, out_shape)


# ============================================================
# Python fallback helpers
# ============================================================


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


# ============================================================
# Operations
# ============================================================


def add(a: Tensor, b: Tensor) -> Tensor:
    if _use_cpp_dispatch(a, b):
        result = _dispatch_cpp("add", a, b)
    else:
        result = _broadcast_binary_op(a, b, lambda x, y: x + y)

    from ..tgir.tracer import record_if_tracing

    record_if_tracing("add", [a, b], result)
    return result


def mul(a: Tensor, b: Tensor) -> Tensor:
    if _use_cpp_dispatch(a, b):
        result = _dispatch_cpp("mul", a, b)
    else:
        result = _broadcast_binary_op(a, b, lambda x, y: x * y)

    from ..tgir.tracer import record_if_tracing

    record_if_tracing("mul", [a, b], result)
    return result


def neg(a: Tensor) -> Tensor:
    if _use_cpp_dispatch(a):
        result = _dispatch_cpp("neg", a)
    else:
        result_data = [-val for val in a.flat_iter()]
        result = _result_tensor(result_data, a)

    from ..tgir.tracer import record_if_tracing

    record_if_tracing("neg", [a], result)
    return result


def sub(a: Tensor, b: Tensor) -> Tensor:
    if _use_cpp_dispatch(a, b):
        result = _dispatch_cpp("sub", a, b)
    else:
        result = _broadcast_binary_op(a, b, lambda x, y: x - y)

    from ..tgir.tracer import record_if_tracing

    record_if_tracing("sub", [a, b], result)
    return result


def div(a: Tensor, b: Tensor) -> Tensor:
    if _use_cpp_dispatch(a, b):
        result = _dispatch_cpp("div", a, b)
    else:
        result = _broadcast_binary_op(a, b, lambda x, y: x / y)

    from ..tgir.tracer import record_if_tracing

    record_if_tracing("div", [a, b], result)
    return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul only supports 2D tensors")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul shape mismatch: {a.shape} @ {b.shape}, inner dimensions must match"
        )

    if _use_cpp_dispatch(a, b):
        result = _dispatch_cpp("matmul", a, b)
    else:
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

        result = _result_tensor(result_data, a, result_shape)

    from ..tgir.tracer import record_if_tracing

    record_if_tracing("matmul", [a, b], result)
    return result


def relu(x: Tensor) -> Tensor:
    if _use_cpp_dispatch(x):
        result = _dispatch_cpp("relu", x)
    else:
        result_data = [val if val > 0 else 0.0 for val in x.flat_iter()]
        result = _result_tensor(result_data, x)

    from ..tgir.tracer import record_if_tracing

    record_if_tracing("relu", [x], result)
    return result
