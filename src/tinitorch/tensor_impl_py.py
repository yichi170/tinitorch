"""Python implementation of TensorImpl for TiniTorch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .device import Device
from .dtype import DType
from .storage import Storage

if TYPE_CHECKING:
    from typing import Self


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


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class TensorImplPy:
    """
    Python implementation of TensorImpl.

    Matches the interface of the C++ TensorImpl for interoperability.
    """

    def __init__(
        self,
        data: list,
        dtype: DType = DType.FLOAT32,
        device: Device | str = "cpu",
    ):
        """
        Create a TensorImpl from nested list.

        Args:
            data: Nested list of numbers.
            dtype: Data type for the tensor.
            device: Device to place tensor on.
        """
        self._dtype = dtype

        if isinstance(device, str):
            device = Device(device)
        if device.type != "cpu":
            raise NotImplementedError("CUDA not yet implemented")
        self._device = device

        self._shape = _infer_shape(data)
        flat_data = _flatten(data)
        self._storage = Storage.from_list(flat_data, self._dtype, self._device)

        self._strides = _compute_strides(self._shape)
        self._offset = 0
        self._numel = _compute_numel(self._shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    def numel(self) -> int:
        return self._numel

    def _compute_flat_index(self, indices: tuple[int, ...]) -> int:
        """Convert multi-dim indices to flat storage index."""
        flat_idx = self._offset
        for i, stride in zip(indices, self._strides):
            flat_idx += i * stride
        return flat_idx

    def __getitem__(self, indices) -> float:
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != self.ndim:
            raise NotImplementedError("Slicing not yet implemented, use full indices")

        flat_idx = self._compute_flat_index(indices)
        return self._storage[flat_idx]

    def __setitem__(self, indices, value: float) -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != self.ndim:
            raise NotImplementedError("Slicing not yet implemented, use full indices")

        flat_idx = self._compute_flat_index(indices)
        self._storage[flat_idx] = value

    def get_flat(self, idx: int) -> float:
        return self._storage[idx]

    def set_flat(self, idx: int, value: float) -> None:
        self._storage[idx] = value

    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory (row-major order)."""
        expected_strides = _compute_strides(self._shape)
        return self._strides == expected_strides and self._offset == 0

    def contiguous(self) -> Self:
        if self.is_contiguous():
            return self

        new_storage = Storage(self.numel(), self._dtype, self._device)
        for i in range(self.numel()):
            indices = []
            remaining = i
            for dim in self._shape:
                indices.append(
                    remaining // _compute_numel(self._shape[len(indices) + 1 :])
                )
                remaining %= _compute_numel(self._shape[len(indices) :])
            new_storage[i] = self[tuple(indices)]

        result = TensorImplPy.__new__(TensorImplPy)
        result._storage = new_storage
        result._shape = self._shape
        result._strides = _compute_strides(self._shape)
        result._offset = 0
        result._dtype = self._dtype
        result._device = self._device
        result._numel = self._numel
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

        result = TensorImplPy.__new__(TensorImplPy)
        result._storage = self._storage
        result._shape = tuple(new_shape)
        result._strides = _compute_strides(tuple(new_shape))
        result._offset = self._offset
        result._dtype = self._dtype
        result._device = self._device
        result._numel = self._numel
        return result

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

        result = TensorImplPy.__new__(TensorImplPy)
        result._storage = self._storage
        result._shape = tuple(new_shape)
        result._strides = tuple(new_strides)
        result._offset = self._offset
        result._dtype = self._dtype
        result._device = self._device
        result._numel = self._numel
        return result

    @property
    def T(self) -> Self:
        if self.ndim < 2:
            return self
        return self.transpose(-2, -1)

    def clone(self) -> Self:
        result = TensorImplPy.__new__(TensorImplPy)
        result._storage = self._storage.copy()
        result._shape = self._shape
        result._strides = _compute_strides(self._shape)
        result._offset = 0
        result._dtype = self._dtype
        result._device = self._device
        result._numel = self._numel
        return result

    def __repr__(self) -> str:
        return (
            f"TensorImplPy(shape={self._shape}, "
            f"dtype={self._dtype}, device={self._device})"
        )
