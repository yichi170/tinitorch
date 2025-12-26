"""Tensor implementation for TiniTorch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .backend import resolve_backend
from .device import Device
from .dtype import DType

if TYPE_CHECKING:
    from typing import Self

# Lazy imports to avoid circular dependencies
_TensorImplPy = None
_TensorImplCpp = None


def _get_impl_class(backend: str):
    global _TensorImplPy, _TensorImplCpp

    if backend == "python":
        if _TensorImplPy is None:
            from .tensor_impl_py import TensorImplPy

            _TensorImplPy = TensorImplPy
        return _TensorImplPy
    elif backend == "cpp":
        if _TensorImplCpp is None:
            from . import _C

            _TensorImplCpp = _C.TensorImpl
        return _TensorImplCpp
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _convert_dtype_for_cpp(dtype: DType):
    from . import _C

    dtype_map = {
        DType.INT32: _C.DType.Int32,
        DType.INT64: _C.DType.Int64,
        DType.FLOAT32: _C.DType.Float32,
        DType.FLOAT64: _C.DType.Float64,
    }
    return dtype_map[dtype]


def _convert_device_for_cpp(device: Device):
    from . import _C

    if device.type == "cpu":
        return _C.Device.CPU
    elif device.type == "cuda":
        return _C.Device.CUDA
    else:
        raise ValueError(f"Unknown device type: {device.type}")


class Tensor:
    """
    TiniTorch Tensor - user-facing tensor class.

    Supports both Python and C++ backends via the `backend` parameter.
    """

    def __init__(
        self,
        data: list | Self,
        dtype: DType | None = DType.FLOAT32,
        device: str | Device = "cpu",
        requires_grad: bool = False,
        backend: Literal["python", "cpp"] | None = None,
    ):
        """
        Args:
            data: Nested list or another Tensor.
            dtype: Data type (default: float32).
            device: Device to place tensor on (default: cpu).
            requires_grad: Whether to track gradients (for autograd).
            backend: Implementation backend ("python", "cpp", or None for auto).
        """
        if isinstance(device, str):
            device = Device(device)
        if device.type != "cpu":
            raise NotImplementedError("CUDA not yet implemented")

        self._backend = resolve_backend(backend)

        if isinstance(data, Tensor):
            if backend is None:
                self._backend = data._backend
            self._impl = data._impl.clone()
        else:
            if self._backend == "cpp":
                cpp_dtype = _convert_dtype_for_cpp(dtype)
                cpp_device = _convert_device_for_cpp(device)
                ImplClass = _get_impl_class("cpp")
                self._impl = ImplClass(data, cpp_dtype, cpp_device)
            else:
                ImplClass = _get_impl_class("python")
                self._impl = ImplClass(data, dtype, device)

        self._dtype = dtype
        self._device = device

        # TODO: support autograd
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None
        self.grad_fn = None

    @property
    def shape(self) -> tuple[int, ...]:
        s = self._impl.shape
        return tuple(s) if not isinstance(s, tuple) else s

    @property
    def strides(self) -> tuple[int, ...]:
        s = self._impl.strides
        return tuple(s) if not isinstance(s, tuple) else s

    @property
    def ndim(self) -> int:
        return self._impl.ndim

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def backend(self) -> str:
        return self._backend

    def numel(self) -> int:
        return self._impl.numel()

    def is_contiguous(self) -> bool:
        return self._impl.is_contiguous()

    def __getitem__(self, indices):
        # Handle single index
        if not isinstance(indices, tuple):
            indices = (indices,)

        # For now, only support full indexing (all dimensions)
        if len(indices) != self.ndim:
            raise NotImplementedError("Slicing not yet implemented, use full indices")

        return self._impl[indices]

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != self.ndim:
            raise NotImplementedError("Slicing not yet implemented, use full indices")

        self._impl[indices] = value

    def to(self, device: str | Device) -> Self:
        device = Device(device) if isinstance(device, str) else device
        if device == self._device:
            return self

        # TODO: Implement CPU <-> CUDA transfer
        raise NotImplementedError("Device transfer not yet implemented")

    def cpu(self) -> Self:
        return self.to("cpu")

    def cuda(self, index: int = 0) -> Self:
        return self.to(f"cuda:{index}")

    def __add__(self, other):
        from .ops import add

        return add(self, other)

    def __mul__(self, other):
        from .ops import mul

        return mul(self, other)

    def __neg__(self):
        from .ops import neg

        return neg(self)

    def __sub__(self, other):
        from .ops import sub

        return sub(self, other)

    def __truediv__(self, other):
        from .ops import div

        return div(self, other)

    def __matmul__(self, other):
        from .ops import matmul

        return matmul(self, other)

    def _wrap_impl(self, new_impl) -> Self:
        """Wrap a new impl in a Tensor, preserving metadata."""
        result = Tensor.__new__(Tensor)
        result._impl = new_impl
        result._backend = self._backend
        result._dtype = self._dtype
        result._device = self._device
        result.requires_grad = self.requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    @staticmethod
    def wrap_impl(new_impl) -> Self:
        result = Tensor.__new__(Tensor)
        result._impl = new_impl
        try:
            from . import _C

            if isinstance(new_impl, _C.TensorImpl):
                result._backend = "cpp"
            else:
                result._backend = "python"
        except ImportError:
            result._backend = "python"
        result._dtype = new_impl.dtype
        result._device = new_impl.device
        result.requires_grad = False
        result.grad = None
        result.grad_fn = None
        return result

    def view(self, *new_shape: int) -> Self:
        if self._backend == "cpp":
            new_impl = self._impl.view(list(new_shape))
        else:
            new_impl = self._impl.view(*new_shape)
        return self._wrap_impl(new_impl)

    def reshape(self, *new_shape: int) -> Self:
        try:
            return self.view(*new_shape)
        except RuntimeError:
            return self.contiguous().view(*new_shape)

    def transpose(self, dim0: int, dim1: int) -> Self:
        new_impl = self._impl.transpose(dim0, dim1)
        return self._wrap_impl(new_impl)

    @property
    def T(self) -> Self:
        if self.ndim < 2:
            return self
        new_impl = self._impl.T
        return self._wrap_impl(new_impl)

    def contiguous(self) -> Self:
        if self.is_contiguous():
            return self
        new_impl = self._impl.contiguous()
        return self._wrap_impl(new_impl)

    def clone(self) -> Self:
        new_impl = self._impl.clone()
        return self._wrap_impl(new_impl)

    def flat_iter(self):
        """Iterate over elements in flat (row-major) order."""
        for i in range(self.numel()):
            indices = self._flat_to_indices(i)
            yield self[indices]

    def _flat_to_indices(self, flat_idx: int) -> tuple:
        """Convert flat index to multi-dim indices."""
        indices = []
        remaining = flat_idx
        for dim_idx in range(len(self.shape)):
            product = 1
            for d in self.shape[dim_idx + 1 :]:
                product *= d
            indices.append(remaining // product)
            remaining %= product
        return tuple(indices)

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, dtype={self._dtype}, "
            f"device={self._device}, backend={self._backend})"
        )

    def __str__(self):
        n = min(10, self.numel())
        preview = [val for val, _ in zip(self.flat_iter(), range(n))]
        suffix = "..." if self.numel() > 10 else ""
        return f"tensor(shape={self.shape}, data={preview}{suffix})"
