"""Activation functions as modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import relu as relu_fn
from .module import Module

if TYPE_CHECKING:
    from ..tensor import Tensor


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu_fn(x)
