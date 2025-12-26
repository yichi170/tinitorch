"""Tests for Storage class."""

import pytest

from tinitorch.dtype import float32
from tinitorch.storage import Storage


def test_storage_creation():
    s = Storage(10, float32)
    assert s.size == 10
    assert s.dtype == float32
    assert s.itemsize == 4  # float32 = 4 bytes
    assert s.nbytes == 40  # 10 * 4 bytes


def test_storage_read_write():
    s = Storage(3, float32)

    s[0] = 1.5
    s[1] = 2.5
    s[2] = 3.5

    assert s[0] == pytest.approx(1.5)
    assert s[1] == pytest.approx(2.5)
    assert s[2] == pytest.approx(3.5)


def test_storage_fill():
    s = Storage(5, float32)
    s.fill(3.14)

    for i in range(5):
        assert s[i] == pytest.approx(3.14)


def test_storage_from_list():
    data = [1.0, 2.0, 3.0, 4.0]
    s = Storage.from_list(data, float32)

    assert s.size == 4
    assert s.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_storage_copy():
    s1 = Storage.from_list([1.0, 2.0, 3.0], float32)
    s2 = s1.copy()

    s1[0] = 999.0

    assert s2[0] == pytest.approx(1.0)


def test_storage_bounds_check():
    s = Storage(3, float32)

    with pytest.raises(IndexError):
        _ = s[3]

    with pytest.raises(IndexError):
        s[-1] = 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
