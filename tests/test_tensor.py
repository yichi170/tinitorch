"""Tests for Tensor class."""

import pytest
import tinitorch as tt


def test_tensor_creation():
    t = tt.Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert t.dtype == tt.float32

    t = tt.Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)

    t = tt.Tensor([1, 2, 3], dtype=tt.int32)
    assert t.dtype == tt.int32


def test_tensor_properties():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.numel() == 6


def test_tensor_device():
    t = tt.Tensor([1, 2, 3], device="cpu")
    assert t.device == tt.Device("cpu")

    # CUDA not yet implemented
    with pytest.raises(NotImplementedError):
        t = tt.Tensor([1, 2, 3], device="cuda")


def test_tensor_data():
    t = tt.Tensor([[1, 2], [3, 4]])
    assert t._data.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert t.shape == (2, 2)


def test_tensor_strides():
    t = tt.Tensor([1, 2, 3])
    assert t.strides == (1,)

    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.strides == (3, 1)

    t = tt.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert t.shape == (2, 2, 2)
    assert t.strides == (4, 2, 1)


def test_tensor_indexing():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    # shape=(2,3), strides=(3,1), storage=[1,2,3,4,5,6]

    assert t[0, 0] == pytest.approx(1.0)
    assert t[0, 1] == pytest.approx(2.0)
    assert t[0, 2] == pytest.approx(3.0)
    assert t[1, 0] == pytest.approx(4.0)
    assert t[1, 1] == pytest.approx(5.0)
    assert t[1, 2] == pytest.approx(6.0)


def test_tensor_indexing_3d():
    t = tt.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # shape=(2,2,2), strides=(4,2,1)

    assert t[0, 0, 0] == pytest.approx(1.0)
    assert t[0, 0, 1] == pytest.approx(2.0)
    assert t[0, 1, 0] == pytest.approx(3.0)
    assert t[1, 0, 0] == pytest.approx(5.0)
    assert t[1, 1, 1] == pytest.approx(8.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
