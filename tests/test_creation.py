"""Tests for tensor creation functions."""

import pytest

import tinitorch as tt


def test_zeros():
    t = tt.zeros(2, 3)
    assert t.shape == (2, 3)
    assert all(v == 0.0 for v in t.flat_iter())


def test_ones():
    t = tt.ones(3, 2)
    assert t.shape == (3, 2)
    assert all(v == 1.0 for v in t.flat_iter())


def test_empty():
    t = tt.empty(4, 4)
    assert t.shape == (4, 4)
    assert t.numel() == 16


def test_randn():
    t = tt.randn(100)
    assert t.shape == (100,)
    values = list(t.flat_iter())
    has_negative = any(v < 0 for v in values)
    has_positive = any(v > 0 for v in values)
    assert has_negative and has_positive


def test_arange():
    t = tt.arange(5)
    assert t.shape == (5,)
    assert list(t.flat_iter()) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_arange_start_end():
    t = tt.arange(2, 7)
    assert t.shape == (5,)
    assert list(t.flat_iter()) == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0])


def test_arange_step():
    t = tt.arange(0, 10, 2)
    assert t.shape == (5,)
    assert list(t.flat_iter()) == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
