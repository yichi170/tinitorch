"""Tensor implementation for TiniTorch."""

from typing import Self, Union, Tuple, Optional, List as ListType


from .dtype import DType, float32
from .device import Device
from .storage import Storage


def _infer_shape(data) -> Tuple[int, ...]:
    if not isinstance(data, list):
        return ()

    shape = [len(data)]
    current = data
    while isinstance(current[0], list):
        shape.append(len(current[0]))
        current = current[0]
    return tuple(shape)


def _flatten(data) -> ListType:
    if not isinstance(data, list):
        return [data]

    result = []
    for item in data:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _compute_numel(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 1
    result = 1
    for dim in shape:
        result *= dim
    return result


def compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
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
        data: Union[list, Storage, Self],
        dtype: Optional[DType] = DType.FLOAT32,
        device: Union[str, Device] = "cpu",
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
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        return self._strides

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def numel(self) -> int:
        return _compute_numel(self._shape)

    def _compute_flat_index(self, indices: Tuple[int, ...]) -> int:
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

    def to(self, device: Union[str, Device]) -> "Tensor":
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

    def __repr__(self):
        return f"Tensor(shape={self._shape}, dtype={self.dtype}, device={self.device})"

    def __str__(self):
        return f"tensor(shape={self._shape}, data={self._data[:10]}{'...' if len(self._data) > 10 else ''})"
