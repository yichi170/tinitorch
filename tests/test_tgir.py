"""Tests for tgir (TiniTorch Graph IR) module."""

import pytest

import tinitorch as tt
from tinitorch.dtype import DType
from tinitorch.tgir import Graph, Node, Value, is_tracing, trace


class TestValue:
    def test_value_creation(self):
        v = Value("x", DType.FLOAT32, (2, 3))
        assert v.name == "x"
        assert v.type == DType.FLOAT32
        assert v.shape == (2, 3)
        assert v.producer is None

    def test_value_str(self):
        v = Value("x", DType.FLOAT32, (2, 3))
        assert str(v) == "%x: float32[2x3]"

    def test_value_repr(self):
        v = Value("x", DType.FLOAT32, (2, 3))
        assert repr(v) == "Value(x, float32, (2, 3))"


class TestNode:
    def test_node_creation(self):
        v1 = Value("a", DType.FLOAT32, (2, 3))
        v2 = Value("b", DType.FLOAT32, (2, 3))
        node = Node("add", [v1, v2], [], {})
        assert node.op == "add"
        assert len(node.inputs) == 2
        assert len(node.outputs) == 0

    def test_node_with_attrs(self):
        v = Value("x", DType.FLOAT32, (3, 4))
        node = Node("conv", [v], [], {"kernel": 3, "stride": 1})
        assert node.attrs["kernel"] == 3
        assert node.attrs["stride"] == 1


class TestGraph:
    def test_graph_creation(self):
        g = Graph("test")
        assert g.name == "test"
        assert g.nodes == []
        assert g.inputs == []
        assert g.outputs == []

    def test_graph_add_input(self):
        g = Graph()
        v = g.add_input("x", DType.FLOAT32, (2, 3))
        assert len(g.inputs) == 1
        assert v.name == "x"
        assert v.type == DType.FLOAT32
        assert v.shape == (2, 3)

    def test_graph_add_node(self):
        g = Graph()
        v1 = g.add_input("a", DType.FLOAT32, (2, 3))
        v2 = g.add_input("b", DType.FLOAT32, (2, 3))
        node = g.add_node("add", [v1, v2], {})
        assert len(g.nodes) == 1
        assert node.op == "add"

    def test_graph_set_output(self):
        g = Graph()
        v = g.add_input("x", DType.FLOAT32, (2, 3))
        g.set_output([v])
        assert len(g.outputs) == 1


class TestTracing:
    def test_is_tracing_false_by_default(self):
        assert is_tracing() is False

    def test_trace_simple_add(self):
        def add_fn(a, b):
            return a + b

        x = tt.randn(2, 3)
        y = tt.randn(2, 3)
        graph = trace(add_fn, x, y)

        assert len(graph.inputs) == 2
        assert len(graph.nodes) == 1
        assert len(graph.outputs) == 1
        assert graph.nodes[0].op == "add"

    def test_trace_binary_ops(self):
        def fn(a, b):
            c = a + b
            d = c * a
            return d - b

        x = tt.randn(2, 2)
        y = tt.randn(2, 2)
        graph = trace(fn, x, y)

        assert len(graph.nodes) == 3
        ops = [n.op for n in graph.nodes]
        assert ops == ["add", "mul", "sub"]

    def test_trace_unary_ops(self):
        def fn(a):
            b = -a
            return tt.relu(b)

        x = tt.randn(3, 3)
        graph = trace(fn, x)

        assert len(graph.inputs) == 1
        assert len(graph.nodes) == 2
        ops = [n.op for n in graph.nodes]
        assert ops == ["neg", "relu"]

    def test_trace_matmul(self):
        def fn(a, b):
            return a @ b

        x = tt.randn(4, 8)
        y = tt.randn(8, 3)
        graph = trace(fn, x, y)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op == "matmul"

    def test_trace_chain(self):
        def fn(x, w1, w2):
            h = x @ w1
            h = tt.relu(h)
            return h @ w2

        x = tt.randn(4, 8)
        w1 = tt.randn(8, 16)
        w2 = tt.randn(16, 4)
        graph = trace(fn, x, w1, w2)

        assert len(graph.inputs) == 3
        assert len(graph.nodes) == 3
        ops = [n.op for n in graph.nodes]
        assert ops == ["matmul", "relu", "matmul"]


class TestTraceNN:
    def test_trace_linear_functional(self):
        def linear(x, weight, bias):
            return x @ weight.T + bias

        x = tt.randn(3, 4)
        w = tt.randn(2, 4)
        b = tt.randn(2)
        graph = trace(linear, x, w, b)

        # x @ W.T + b -> transpose + matmul + add
        assert len(graph.inputs) == 3
        assert len(graph.nodes) == 3
        ops = [n.op for n in graph.nodes]
        assert "transpose" in ops
        assert "matmul" in ops
        assert "add" in ops

    def test_trace_mlp_functional(self):
        def mlp(x, w1, b1, w2, b2):
            h = x @ w1.T + b1
            h = tt.relu(h)
            return h @ w2.T + b2

        x = tt.randn(2, 4)
        w1 = tt.randn(3, 4)
        b1 = tt.randn(3)
        w2 = tt.randn(2, 3)
        b2 = tt.randn(2)

        graph = trace(mlp, x, w1, b1, w2, b2)

        assert len(graph.inputs) == 5
        assert len(graph.outputs) == 1
        # (transpose + matmul + add) + relu + (transpose + matmul + add) = 7 nodes
        assert len(graph.nodes) == 7

    def test_trace_module(self):
        import tinitorch.nn as nn

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 3)  # weight (3,4) + bias (3,) = 2 params
                self.fc2 = nn.Linear(3, 2)  # weight (2,3) + bias (2,) = 2 params

            def forward(self, x):
                x = tt.relu(self.fc1(x))
                return self.fc2(x)

        model = MLP()
        x = tt.randn(2, 4)
        graph = trace(model, x)

        # Graph inputs: 4 parameters + 1 user input = 5
        assert len(graph.inputs) == 5
        assert len(graph.outputs) == 1
        # fc1: transpose + matmul + add = 3
        # relu = 1
        # fc2: transpose + matmul + add = 3
        # Total = 7 nodes
        assert len(graph.nodes) == 7


class TestGraphStructure:
    def test_value_producer_linked(self):
        def fn(a, b):
            return a + b

        x = tt.randn(2, 2)
        y = tt.randn(2, 2)
        graph = trace(fn, x, y)

        output_value = graph.outputs[0]
        assert output_value.producer is not None
        assert output_value.producer.op == "add"

    def test_node_outputs_linked(self):
        def fn(a, b):
            return a + b

        x = tt.randn(2, 2)
        y = tt.randn(2, 2)
        graph = trace(fn, x, y)

        node = graph.nodes[0]
        assert len(node.outputs) == 1
        assert node.outputs[0] == graph.outputs[0]

    def test_input_no_producer(self):
        def fn(a):
            return tt.relu(a)

        x = tt.randn(2, 2)
        graph = trace(fn, x)

        assert graph.inputs[0].producer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
