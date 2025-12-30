"""Linear layer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..creation import uniform
from .module import Module

if TYPE_CHECKING:
    from ..tensor import Tensor


class Linear(Module):
    """Applies a linear transformation: y = x @ W.T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform initialization
        # weight ~ U(-bound, bound) where bound = 1 / sqrt(in_features)
        bound = 1.0 / math.sqrt(in_features)

        self.weight = self.add_parameter(
            "weight", uniform(out_features, in_features, low=-bound, high=bound)
        )

        if bias:
            self.bias = self.add_parameter(
                "bias", uniform(out_features, low=-bound, high=bound)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y
