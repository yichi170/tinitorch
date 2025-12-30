"""Tests for broadcasting functionality."""

import pytest

import tinitorch as tt


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_broadcast_add_1d(backend):
    """Test (2, 3) + (3,) -> (2, 3)"""
    a = tt.Tensor([[1, 2, 3], [4, 5, 6]], backend=backend)
    b = tt.Tensor([10, 20, 30], backend=backend)
    c = a + b

    assert c.shape == (2, 3)
    assert c[0, 0] == pytest.approx(11.0)
    assert c[0, 1] == pytest.approx(22.0)
    assert c[1, 2] == pytest.approx(36.0)


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_broadcast_add_scalar_like(backend):
    """Test (3,) + (1,) -> (3,)"""
    a = tt.Tensor([1, 2, 3], backend=backend)
    b = tt.Tensor([10], backend=backend)
    c = a + b

    assert c.shape == (3,)
    assert c[0] == pytest.approx(11.0)
    assert c[1] == pytest.approx(12.0)
    assert c[2] == pytest.approx(13.0)


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_broadcast_sub(backend):
    """Test (2, 3) - (3,) -> (2, 3)"""
    a = tt.Tensor([[10, 20, 30], [40, 50, 60]], backend=backend)
    b = tt.Tensor([1, 2, 3], backend=backend)
    c = a - b

    assert c.shape == (2, 3)
    assert c[0, 0] == pytest.approx(9.0)
    assert c[1, 2] == pytest.approx(57.0)


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_broadcast_mul(backend):
    """Test (2, 3) * (3,) -> (2, 3)"""
    a = tt.Tensor([[1, 2, 3], [4, 5, 6]], backend=backend)
    b = tt.Tensor([2, 2, 2], backend=backend)
    c = a * b

    assert c.shape == (2, 3)
    assert c[0, 0] == pytest.approx(2.0)
    assert c[1, 2] == pytest.approx(12.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
