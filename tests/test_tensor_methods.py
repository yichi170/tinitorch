"""Tests for new tensor features: view, transpose, contiguous, etc."""

import pytest

import tinitorch as tt


def test_view():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)

    v = t.view(3, 2)
    assert v.shape == (3, 2)

    # Verify values match (shared storage)
    assert list(v.flat_iter()) == list(t.flat_iter())


def test_view_minus_one():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    v = t.view(-1)
    assert v.shape == (6,)

    v = t.view(3, -1)
    assert v.shape == (3, 2)


def test_reshape():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    r = t.reshape(6)
    assert r.shape == (6,)


def test_transpose():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    # shape=(2,3), strides=(3,1)

    tr = t.transpose(0, 1)
    assert tr.shape == (3, 2)
    assert tr.strides == (1, 3)

    assert tr[0, 0] == pytest.approx(1.0)
    assert tr[0, 1] == pytest.approx(4.0)
    assert tr[2, 0] == pytest.approx(3.0)
    assert tr[2, 1] == pytest.approx(6.0)


def test_transpose_T():
    t = tt.Tensor([[1, 2], [3, 4]])
    tr = t.T
    assert tr.shape == (2, 2)
    assert tr[0, 1] == pytest.approx(3.0)
    assert tr[1, 0] == pytest.approx(2.0)


def test_is_contiguous():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.is_contiguous()

    tr = t.transpose(0, 1)
    assert not tr.is_contiguous()


def test_contiguous():
    t = tt.Tensor([[1, 2, 3], [4, 5, 6]])
    tr = t.transpose(0, 1)
    assert not tr.is_contiguous()

    c = tr.contiguous()
    assert c.is_contiguous()
    assert c.shape == (3, 2)
    assert c[0, 0] == pytest.approx(1.0)
    assert c[2, 1] == pytest.approx(6.0)


def test_clone():
    t = tt.Tensor([1, 2, 3])
    c = t.clone()

    # Verify independent copy with same values
    assert list(c.flat_iter()) == list(t.flat_iter())
    # Modifying clone shouldn't affect original
    c[0] = 99.0
    assert t[0] == pytest.approx(1.0)


def test_setitem():
    t = tt.Tensor([[1, 2], [3, 4]])
    t[0, 0] = 99.0
    assert t[0, 0] == pytest.approx(99.0)

    t[1, 1] = -5.0
    assert t[1, 1] == pytest.approx(-5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
