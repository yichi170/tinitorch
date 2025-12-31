"""Graph IR for TiniTorch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..dtype import DType


@dataclass
class Value:
    name: str
    type: DType
    producer: Node | None = None  # The node that produced this value

    def __str__(self) -> str:
        return f"%{self.name}({self.type})"

    def __repr__(self) -> str:
        return f"Value({self.name}, {self.type})"


@dataclass
class Node:
    op: str
    inputs: list[Value]
    outputs: list[Value]
    attrs: dict[str, Any]

    def __repr__(self) -> str:
        inputs = ", ".join(str(i) for i in self.inputs)
        outputs = ", ".join(str(o) for o in self.outputs)
        return f"!{self.op}: ins=[{inputs}], outs=[{outputs}], attrs={self.attrs}"


class Graph:
    def __init__(self, name: str = "main_graph"):
        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []

    def add_input(self, name: str, type: DType) -> Value:
        value = Value(name, type)
        self.inputs.append(value)
        return value

    def set_output(self, values: Sequence[Value]) -> None:
        self.outputs = list(values)

    def add_node(self, op: str, inputs: Sequence[Value], attrs: dict[str, Any]) -> Node:
        node = Node(op=op, inputs=list(inputs), outputs=[], attrs=attrs)
        self.nodes.append(node)
        return node

    def __repr__(self) -> str:
        lines = [f"Graph({self.name}): {{"]
        lines.append(f"    Inputs: {self.inputs}")
        lines.append(f"    Outputs: {self.outputs}")
        lines.append("    Operations:")
        for node in self.nodes:
            lines.append(f"        {node!r}")
        lines.append("}")
        return "\n".join(lines)
