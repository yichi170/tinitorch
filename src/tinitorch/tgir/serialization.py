"""Graph serialization to JSON for compiler import."""

from __future__ import annotations

import json
from typing import Any

from .ir import Graph, Node, Value


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    return {
        "name": graph.name,
        "inputs": [_value_to_dict(v) for v in graph.inputs],
        "outputs": [v.name for v in graph.outputs],
        "nodes": [_node_to_dict(n) for n in graph.nodes],
    }


def _value_to_dict(value: Value) -> dict[str, Any]:
    return {
        "name": value.name,
        "dtype": str(value.type),
        "shape": list(value.shape),
    }


def _node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "op": node.op,
        "inputs": [v.name for v in node.inputs],
        "outputs": [_value_to_dict(v) for v in node.outputs],
        "attrs": node.attrs,
    }


def graph_to_json(graph: Graph, indent: int | None = 2) -> str:
    return json.dumps(graph_to_dict(graph), indent=indent)


def save_graph(graph: Graph, path: str) -> None:
    with open(path, "w") as f:
        f.write(graph_to_json(graph))


def to_json(graph: Graph, indent: int | None = 2) -> str:
    return graph_to_json(graph, indent)


def export(graph: Graph, path: str) -> None:
    save_graph(graph, path)
