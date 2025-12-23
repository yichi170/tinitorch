"""Tensor implementation for TiniTorch."""

from typing import Self

from .device import Device
from .dtype import DType
from .storage import Storage


def _infer_shape(data) -> tuple[int, ...]:
    if not isinstance(data, list):
        return ()

    shape = [len(data)]
    current = data
    while isinstance(current[0], list):
        shape.append(len(current[0]))
        current = current[0]
    return tuple(shape)


def _flatten(data) -> list:
    if not isinstance(data, list):
        return [data]

    result = []
    for item in data:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _compute_numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    result = 1
    for dim in shape:
        result *= dim
    return result


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class Tensor:
    """
    TiniTorch Tensor - user-facing tensor class.
    """

    def __init__(
        self,
        data: list | Storage | Self,
        dtype: DType | None = DType.FLOAT32,
        device: str | Device = "cpu",
        requires_grad: bool = False,
    ):
        """
        Args:
            data: Nested list or another Tensor
            dtype: Data type (default: float32)
            device: Device to place tensor on
            requires_grad: Whether to track gradients (for autograd)
        """

        self.dtype = dtype

        if isinstance(device, str):
            device = Device(device)
        if device.type != "cpu":
            raise NotImplementedError("CUDA not yet implemented")

        self.device = device

        if isinstance(data, Tensor):
            self._data = data._data.copy()
            self._shape = data._shape
        else:
            self._shape = _infer_shape(data)
            flat_data = _flatten(data)
            self._data = Storage.from_list(flat_data, self.dtype, self.device)

        self._strides = compute_strides(self._shape)
        self._offset = 0  # Offset into storage (for views/slices)

        # TODO: support autograd
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def numel(self) -> int:
        return _compute_numel(self._shape)

    def _compute_flat_index(self, indices: tuple[int, ...]) -> int:
        """Convert multi-dim indices to flat storage index."""
        flat_idx = self._offset
        for i, stride in zip(indices, self._strides):
            flat_idx += i * stride
        return flat_idx

    def __getitem__(self, indices):
        # Handle single index
        if not isinstance(indices, tuple):
            indices = (indices,)

        # For now, only support full indexing (all dimensions)
        if len(indices) != self.ndim:
            raise NotImplementedError("Slicing not yet implemented, use full indices")

        flat_idx = self._compute_flat_index(indices)
        return self._data[flat_idx]

    def to(self, device: str | Device) -> "Tensor":
        device = Device(device) if isinstance(device, str) else device
        if device == self.device:
            return self

        # TODO: Implement CPU <-> CUDA transfer in Phase 3
        raise NotImplementedError("Device transfer not yet implemented")

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def cuda(self, index: int = 0) -> "Tensor":
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

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != self.ndim:
            raise NotImplementedError("Slicing not yet implemented, use full indices")

        flat_idx = self._compute_flat_index(indices)
        self._data[flat_idx] = value

    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory (row-major order)."""
        expected_strides = compute_strides(self._shape)
        return self._strides == expected_strides and self._offset == 0

    def contiguous(self) -> Self:
        """Return contiguous tensor (copy if necessary)."""
        if self.is_contiguous():
            return self

        new_storage = Storage(self.numel(), self.dtype, self.device)
        for i in range(self.numel()):
            indices = []
            remaining = i
            for dim in self._shape:
                indices.append(
                    remaining // _compute_numel(self._shape[len(indices) + 1 :])
                )
                remaining %= _compute_numel(self._shape[len(indices) :])
            new_storage[i] = self[tuple(indices)]

        result = Tensor.__new__(Tensor)
        result._data = new_storage
        result._shape = self._shape
        result._strides = compute_strides(self._shape)
        result._offset = 0
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    def view(self, *new_shape: int) -> Self:
        if not self.is_contiguous():
            raise RuntimeError(
                "view requires contiguous tensor, call .contiguous() first"
            )

        # Handle -1 dimension
        new_shape = list(new_shape)
        neg_idx = None
        known_numel = 1
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if neg_idx is not None:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_numel *= dim

        if neg_idx is not None:
            new_shape[neg_idx] = self.numel() // known_numel

        # Validate total elements
        new_numel = _compute_numel(tuple(new_shape))
        if new_numel != self.numel():
            raise ValueError(
                f"Cannot reshape tensor of {self.numel()} elements to {new_shape}"
            )

        result = Tensor.__new__(Tensor)
        result._data = self._data
        result._shape = tuple(new_shape)
        result._strides = compute_strides(tuple(new_shape))
        result._offset = self._offset
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    def reshape(self, *new_shape: int) -> Self:
        try:
            return self.view(*new_shape)
        except RuntimeError:
            return self.contiguous().view(*new_shape)

    def transpose(self, dim0: int, dim1: int) -> Self:
        if dim0 < 0:
            dim0 = self.ndim + dim0
        if dim1 < 0:
            dim1 = self.ndim + dim1

        if dim0 >= self.ndim or dim1 >= self.ndim:
            raise IndexError(f"Dimension out of range for {self.ndim}D tensor")

        # Swap shape and strides
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

        result = Tensor.__new__(Tensor)
        result._data = self._data
        result._shape = tuple(new_shape)
        result._strides = tuple(new_strides)
        result._offset = self._offset
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    @property
    def T(self) -> Self:
        if self.ndim < 2:
            return self
        return self.transpose(-2, -1)

    def clone(self) -> Self:
        result = Tensor.__new__(Tensor)
        result._data = self._data.copy()
        result._shape = self._shape
        result._strides = compute_strides(self._shape)
        result._offset = 0
        result.dtype = self.dtype
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    def __repr__(self):
        return f"Tensor(shape={self._shape}, dtype={self.dtype}, device={self.device})"

    def __str__(self):
        preview = self._data[:10]
        suffix = "..." if len(self._data) > 10 else ""
        return f"tensor(shape={self._shape}, data={preview}{suffix})"
