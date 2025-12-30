"""Neural network module base class."""

from __future__ import annotations

from collections.abc import Iterator

from ..tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Subclasses should override forward() to define computation.
    """

    def __init__(self):
        # Use object.__setattr__ so self.__setattr__ is not called here.
        # This prevents problems if self.__setattr__ is changed in the future.
        # In this code, this behaves the same as self._parameters = {}.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value: Tensor | Module):
        if isinstance(value, Tensor):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)

        object.__setattr__(self, name, value)

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

    def _extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra = self._extra_repr()
        if not self._modules:
            if extra:
                return f"{self.__class__.__name__}({extra})"
            return f"{self.__class__.__name__}()"

        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            mod_repr = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_repr}")
        lines.append(")")
        return ",\n".join(lines)
