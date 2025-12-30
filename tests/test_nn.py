"""Tests for nn module API."""

import pytest

import tinitorch as tt
import tinitorch.nn as nn


class TestModule:
    """Tests for nn.Module base class."""

    def test_module_parameters_empty(self):
        module = nn.Module()
        assert list(module.parameters()) == []

    def test_module_modules_empty(self):
        module = nn.Module()
        assert list(module.modules()) == []

    def test_module_forward_not_implemented(self):
        module = nn.Module()
        with pytest.raises(NotImplementedError):
            module(tt.Tensor([1, 2, 3]))


class TestLinear:
    """Tests for nn.Linear layer."""

    def test_linear_creation(self):
        linear = nn.Linear(10, 5)
        assert linear.weight.shape == (5, 10)
        assert linear.bias.shape == (5,)

    def test_linear_forward_shape(self):
        linear = nn.Linear(10, 5)
        x = tt.randn(3, 10)
        y = linear(x)
        assert y.shape == (3, 5)

    def test_linear_no_bias(self):
        linear = nn.Linear(10, 5, bias=False)
        assert linear.weight.shape == (5, 10)
        assert linear.bias is None

    def test_linear_parameters(self):
        linear = nn.Linear(10, 5)
        params = list(linear.parameters())
        assert len(params) == 2

    def test_linear_no_bias_parameters(self):
        linear = nn.Linear(10, 5, bias=False)
        params = list(linear.parameters())
        assert len(params) == 1


class TestReLU:
    """Tests for nn.ReLU activation."""

    def test_relu_forward(self):
        relu = nn.ReLU()
        x = tt.Tensor([-1, 0, 1, 2])
        y = relu(x)
        assert y[0] == pytest.approx(0.0)
        assert y[1] == pytest.approx(0.0)
        assert y[2] == pytest.approx(1.0)
        assert y[3] == pytest.approx(2.0)


class TestSequential:
    """Tests for nn.Sequential container."""

    def test_sequential_forward(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )
        x = tt.randn(3, 10)
        y = model(x)
        assert y.shape == (3, 2)

    def test_sequential_parameters(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        )
        params = list(model.parameters())
        assert len(params) == 4

    def test_sequential_modules(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        modules = list(model.modules())
        assert len(modules) == 2


class TestMLP:
    """Integration test: MLP forward pass."""

    def test_mlp_forward(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = self.add_module("fc1", nn.Linear(4, 3))
                self.fc2 = self.add_module("fc2", nn.Linear(3, 2))

            def forward(self, x):
                x = tt.relu(self.fc1(x))
                return self.fc2(x)

        model = MLP()
        x = tt.randn(2, 4)
        y = model(x)
        assert y.shape == (2, 2)

    def test_mlp_parameters(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = self.add_module("fc1", nn.Linear(4, 3))
                self.fc2 = self.add_module("fc2", nn.Linear(3, 2))

            def forward(self, x):
                return x

        model = MLP()
        params = list(model.parameters())
        assert len(params) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
