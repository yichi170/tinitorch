"""Mathematical operations."""

from ..tensor import Tensor
from ..storage import Storage


def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise addition with broadcasting.

    TODO: Implement in C++ with proper dispatcher
    TODO: Implement proper broadcasting
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_storage = Storage(a.numel(), a.dtype, a.device)

    for i in range(a.numel()):
        result_storage[i] = a._data[i] + b._data[i]

    result = Tensor.__new__(Tensor)
    result._data = result_storage
    result._shape = a.shape
    result.dtype = a.dtype
    result.device = a.device
    result.requires_grad = False
    result.grad = None
    result.grad_fn = None

    return result


def mul(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise multiplication with broadcasting.

    TODO: Implement in C++ with proper dispatcher
    TODO: Implement proper broadcasting
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Broadcasting not yet implemented."
        )

    result_storage = Storage(a.numel(), a.dtype, a.device)

    for i in range(a.numel()):
        result_storage[i] = a._data[i] * b._data[i]

    result = Tensor.__new__(Tensor)
    result._data = result_storage
    result._shape = a.shape
    result.dtype = a.dtype
    result.device = a.device
    result.requires_grad = False
    result.grad = None
    result.grad_fn = None

    return result
