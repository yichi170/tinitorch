"""Tests for tensor operations."""

import pytest

import tinitorch as tt


def test_add():
    """Test element-wise addition."""
    a = tt.Tensor([1, 2, 3])
    b = tt.Tensor([4, 5, 6])
    c = tt.add(a, b)

    assert c._data.tolist() == pytest.approx([5.0, 7.0, 9.0])
    assert c.shape == (3,)

    c = a + b
    assert c._data.tolist() == pytest.approx([5.0, 7.0, 9.0])


def test_mul():
    """Test element-wise multiplication."""
    a = tt.Tensor([1, 2, 3])
    b = tt.Tensor([4, 5, 6])
    c = tt.mul(a, b)

    assert c._data.tolist() == pytest.approx([4.0, 10.0, 18.0])
    assert c.shape == (3,)

    c = a * b
    assert c._data.tolist() == pytest.approx([4.0, 10.0, 18.0])


def test_add_2d():
    """Test 2D tensor addition."""
    a = tt.Tensor([[1, 2], [3, 4]])
    b = tt.Tensor([[5, 6], [7, 8]])
    c = a + b

    assert c.shape == (2, 2)
    assert c._data.tolist() == pytest.approx([6.0, 8.0, 10.0, 12.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
