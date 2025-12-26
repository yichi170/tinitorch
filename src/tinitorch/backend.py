"""Backend utilities for TiniTorch."""

from __future__ import annotations

from typing import Literal

BackendType = Literal["python", "cpp"]

_CPP_AVAILABLE = False
try:
    from . import _C  # noqa: F401

    _CPP_AVAILABLE = True
except ImportError:
    pass

AVAILABLE_BACKENDS: list[BackendType] = ["python"]
if _CPP_AVAILABLE:
    AVAILABLE_BACKENDS.append("cpp")


def get_available_backends() -> list[BackendType]:
    return AVAILABLE_BACKENDS.copy()


def is_cpp_available() -> bool:
    return _CPP_AVAILABLE


def resolve_backend(backend: BackendType | None) -> BackendType:
    if backend is None:
        return "cpp" if _CPP_AVAILABLE else "python"

    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' not available. Available: {AVAILABLE_BACKENDS}"
        )

    return backend
