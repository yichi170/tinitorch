"""Storage implementation for TiniTorch.

Storage is the low-level contiguous memory buffer that backs Tensor data.
Multiple Tensors can share the same Storage (views).
"""

import struct
from typing import Union

from .dtype import DType
from .device import Device


# Bytes per element for each dtype
DTYPE_SIZES = {
    DType.FLOAT32: 4,
    DType.FLOAT64: 8,
    DType.INT32: 4,
    DType.INT64: 8,
}

# struct format codes for each dtype
DTYPE_FORMATS = {
    DType.FLOAT32: "f",
    DType.FLOAT64: "d",
    DType.INT32: "i",
    DType.INT64: "q",
}


class Storage:
    """
    Contiguous memory buffer for tensor data.

    This is the low-level storage that holds raw bytes.
    Tensors reference Storage with shape/strides/offset metadata.
    """

    def __init__(self, size: int, dtype: DType, device: Device = None):
        """
        Allocate storage for `size` elements of type `dtype`.

        Args:
            size: Number of elements (not bytes)
            dtype: Data type of elements
            device: Device for storage (CPU only for now)
        """
        if device is None:
            device = Device("cpu")

        if device.type != "cpu":
            raise NotImplementedError("Only CPU storage implemented")

        self.size = size
        self.dtype = dtype
        self.device = device

        self._itemsize = DTYPE_SIZES[dtype]
        self._nbytes = size * self._itemsize
        self._data = bytearray(self._nbytes)
        self._format = DTYPE_FORMATS[dtype]

    @property
    def itemsize(self) -> int:
        return self._itemsize

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __getitem__(self, idx: int) -> Union[int, float]:
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for size {self.size}")

        offset = idx * self._itemsize
        # Unpack single value from bytes
        (value,) = struct.unpack_from(self._format, self._data, offset)
        return value

    def __setitem__(self, idx: int, value: Union[int, float]):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for size {self.size}")

        offset = idx * self._itemsize
        # Pack value into bytes
        struct.pack_into(self._format, self._data, offset, value)

    def fill(self, value: Union[int, float]):
        for i in range(self.size):
            self[i] = value

    def copy(self) -> "Storage":
        new_storage = Storage(self.size, self.dtype, self.device)
        new_storage._data = bytearray(self._data)
        return new_storage

    def tolist(self) -> list:
        return [self[i] for i in range(self.size)]

    @classmethod
    def from_list(cls, data: list, dtype: DType, device: Device = None) -> "Storage":
        storage = cls(len(data), dtype, device)
        for i, value in enumerate(data):
            storage[i] = value
        return storage

    def __repr__(self):
        preview = self.tolist()[:8]
        suffix = "..." if self.size > 8 else ""
        return f"Storage(size={self.size}, dtype={self.dtype}, data={preview}{suffix})"

    def __len__(self):
        return self.size
