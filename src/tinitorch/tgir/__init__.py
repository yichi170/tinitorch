"""TiniTorch Graph IR (TGIR) - Computation graph representation."""

from .ir import Graph, Node, Value
from .tracer import is_tracing, record_if_tracing, trace

__all__ = [
    # IR
    "Graph",
    "Node",
    "Value",
    # Tracing
    "trace",
    "is_tracing",
    "record_if_tracing",
]
