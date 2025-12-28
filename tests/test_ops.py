"""Tests for tensor operations."""

import pytest

import tinitorch as tt


def test_C_add():
    a = tt.Tensor([1, 2, 3])
    b = tt.Tensor([4, 5, 6])
    c = tt._C.dispatch("add", tt._C.Device.CPU, tt._C.DType.Float32, [a._impl, b._impl])
    c = tt.Tensor.wrap_impl(c)

    assert list(c.flat_iter()) == pytest.approx([5.0, 7.0, 9.0])
    assert c.shape == (3,)


def test_add():
    """Test element-wise addition."""
    a = tt.Tensor([1, 2, 3])
    b = tt.Tensor([4, 5, 6])
    c = tt.add(a, b)

    assert list(c.flat_iter()) == pytest.approx([5.0, 7.0, 9.0])
    assert c.shape == (3,)

    c = a + b
    assert list(c.flat_iter()) == pytest.approx([5.0, 7.0, 9.0])


def test_add_2d_T():
    a = tt.Tensor([[1, 2], [3, 4]])
    b = a.T
    c = a + b

    assert list(c.flat_iter()) == pytest.approx([2.0, 5.0, 5.0, 8.0])

    a[0, 0] = 3
    assert b[0, 0] == 3


def test_mul():
    """Test element-wise multiplication."""
    a = tt.Tensor([1, 2, 3])
    b = tt.Tensor([4, 5, 6])
    c = tt.mul(a, b)

    assert list(c.flat_iter()) == pytest.approx([4.0, 10.0, 18.0])
    assert c.shape == (3,)

    c = a * b
    assert list(c.flat_iter()) == pytest.approx([4.0, 10.0, 18.0])


def test_add_2d():
    """Test 2D tensor addition."""
    a = tt.Tensor([[1, 2], [3, 4]])
    b = tt.Tensor([[5, 6], [7, 8]])
    c = a + b

    assert c.shape == (2, 2)
    assert list(c.flat_iter()) == pytest.approx([6.0, 8.0, 10.0, 12.0])


def test_neg():
    t = tt.Tensor([1, -2, 3])
    r = -t
    assert list(r.flat_iter()) == pytest.approx([-1.0, 2.0, -3.0])


def test_sub():
    a = tt.Tensor([5, 4, 3])
    b = tt.Tensor([1, 2, 3])
    c = a - b
    assert list(c.flat_iter()) == pytest.approx([4.0, 2.0, 0.0])


def test_div():
    a = tt.Tensor([6, 8, 10])
    b = tt.Tensor([2, 4, 5])
    c = a / b
    assert list(c.flat_iter()) == pytest.approx([3.0, 2.0, 2.0])


def test_matmul():
    a = tt.Tensor([[1, 2], [3, 4]])
    b = tt.Tensor([[5, 6], [7, 8]])
    c = a @ b

    assert c.shape == (2, 2)
    # [[1, 2]     [[5, 6]
    #  [3, 4]]  @  [7, 8]]
    # [[19, 22], [43, 50]]
    assert c[0, 0] == pytest.approx(19.0)
    assert c[0, 1] == pytest.approx(22.0)
    assert c[1, 0] == pytest.approx(43.0)
    assert c[1, 1] == pytest.approx(50.0)


def test_matmul_different_shapes():
    a = tt.Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    b = tt.Tensor([[1], [2], [3]])  # 3x1
    c = a @ b

    assert c.shape == (2, 1)
    assert c[0, 0] == pytest.approx(14.0)
    assert c[1, 0] == pytest.approx(32.0)


def test_relu():
    t = tt.Tensor([-2, -1, 0, 1, 2])
    r = tt.relu(t)
    assert list(r.flat_iter()) == pytest.approx([0.0, 0.0, 0.0, 1.0, 2.0])


def test_mlp_forward():
    """Test a simple MLP forward pass: z = relu(x @ W + b)"""
    x = tt.Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    W = tt.Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3x2
    b = tt.Tensor([[0.1, 0.2], [0.1, 0.2]])  # 2x2 (broadcast TODO)

    z = tt.relu(x @ W + b)

    assert z.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert z[i, j] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
