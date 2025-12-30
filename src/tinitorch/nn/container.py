"""Container modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .module import Module

if TYPE_CHECKING:
    from ..tensor import Tensor


class Sequential(Module):
    """A sequential container that runs modules in order."""

    def __init__(self, *modules: Module):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
