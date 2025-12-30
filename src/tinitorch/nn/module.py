"""Neural network module base class."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Subclasses should override forward() to define computation.
    Use add_parameter() and add_module() to register components.
    """

    def __init__(self):
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, Module] = {}

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def add_parameter(self, name: str, param: Tensor) -> Tensor:
        self._parameters[name] = param
        return param

    def add_module(self, name: str, module: Module) -> Module:
        self._modules[name] = module
        return module

    def parameters(self) -> Iterator[Tensor]:
        yield from self._parameters.values()

        for module in self._modules.values():
            yield from module.parameters()

    def modules(self) -> Iterator[Module]:
        yield from self._modules.values()
