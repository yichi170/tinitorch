"""Tracing mechanism to capture computation graphs."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from .ir import Graph, Value

if TYPE_CHECKING:
    from ..nn import Module
    from ..tensor import Tensor

# ============================================================
# Tracing Context
# ============================================================

_trace_context: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "trace_context", default=None
)


def is_tracing() -> bool:
    return _trace_context.get() is not None


def get_trace_context() -> TraceContext | None:
    return _trace_context.get()


class TraceContext:
    """Holds state during graph tracing.

    Attributes:
        Graph: The computation graph being built
        _tensor_to_value: Dict[id(Tensor), Value]
        _value_counter: Counter for generating unique value names
    """

    def __init__(self, graph_name: str = "traced"):
        self.graph = Graph(name=graph_name)
        self._tensor_to_value: dict[int, Value] = {}
        self._value_counter = 0

    def _next_value_name(self, prefix: str = "v") -> str:
        name = f"{prefix}{self._value_counter}"
        self._value_counter += 1
        return name

    def register_input(self, tensor: Tensor) -> Value:
        name = f"arg{len(self.graph.inputs)}"
        value = self.graph.add_input(name, tensor.dtype)
        self._tensor_to_value[id(tensor)] = value
        return value

    def get_value(self, tensor: Tensor) -> Value | None:
        return self._tensor_to_value.get(id(tensor))

    def record_op(
        self,
        op: str,
        inputs: Sequence[Tensor],
        output: Tensor,
        attrs: dict | None = None,
    ) -> Value:
        input_values = []
        for t in inputs:
            v = self._tensor_to_value.get(id(t))
            if v is None:
                raise RuntimeError(
                    "Tensor not found in trace context. "
                    "Did you create a tensor inside the traced function?"
                )
            input_values.append(v)

        output_name = self._next_value_name()
        output_value = Value(name=output_name, type=output.dtype)

        node = self.graph.add_node(op, input_values, attrs or {})

        output_value.producer = node
        node.outputs.append(output_value)

        self._tensor_to_value[id(output)] = output_value

        return output_value


# ============================================================
# Tracing API
# ============================================================


def trace(target: Callable | Module, *example_inputs: Tensor) -> Graph:
    """Trace a function or module to capture its computation graph.

    Args:
        target: A callable (function) or nn.Module to trace
        example_inputs: Example input tensors for shape/dtype inference

    Returns:
        Graph representing the traced computation

    Example:
        ```python
        import tinitorch as tt
        from tinitorch.tgir import trace

        def add_fn(a, b):
            return a + b

        x, y = tt.randn(2, 3), tt.randn(2, 3)
        graph = trace(add_fn, x, y)

        model = nn.Linear(4, 2)
        graph = trace(model, tt.randn(3, 4))
        ```

    Note:
        Python control flow is not supported.
    """

    from ..nn import Module

    if isinstance(target, Module):
        return _trace_module(target, *example_inputs)
    elif callable(target):
        return _trace_callable(target, *example_inputs)
    else:
        raise TypeError(f"Expected callable or nn.Module, got {type(target).__name__}")


def _trace_callable(fn: Callable[..., Tensor], *example_inputs: Tensor) -> Graph:
    ctx = TraceContext()

    for tensor in example_inputs:
        ctx.register_input(tensor)

    return _execute_with_tracing(ctx, fn, *example_inputs)


def _trace_module(module: Module, *example_inputs: Tensor) -> Graph:
    ctx = TraceContext()

    for param in module.parameters():
        ctx.register_input(param)

    for tensor in example_inputs:
        ctx.register_input(tensor)

    return _execute_with_tracing(ctx, module, *example_inputs)


def _execute_with_tracing(
    ctx: TraceContext,
    fn: Callable,
    *inputs: Tensor,
) -> Graph:
    """Execute a function with tracing enabled and finalize the graph."""
    token = _trace_context.set(ctx)
    try:
        output = fn(*inputs)
    finally:
        _trace_context.reset(token)

    if output is not None:
        out_value = ctx.get_value(output)
        if out_value is not None:
            ctx.graph.set_output([out_value])

    return ctx.graph


# ============================================================
# Op Recording Hook
# ============================================================


def record_if_tracing(
    op: str,
    inputs: Sequence[Tensor],
    output: Tensor,
    attrs: dict | None = None,
) -> None:
    ctx = get_trace_context()
    if ctx is not None:
        ctx.record_op(op, inputs, output, attrs)
