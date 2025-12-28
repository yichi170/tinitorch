# TiniTorch

TiniTorch is a tiny neural network framework.

## Quick start
[`uv`](https://docs.astral.sh/uv/) can be installed via:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
# Install dev deps
uv sync

# Run tests
uv run pytest
```

## Optional: build the C++ extension
```bash
cmake -S . -B build \
  -DPython_EXECUTABLE=$(uv run which python) \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build
```

## Tiny usage sample
```python
import tinitorch as tt

a = tt.Tensor([[1, -4], [-3, 4]])
b = tt.ones(2, 2)
c = tt.relu(a + b)
print(c)
```
