---
title: Installation
description: Install greyhound-kernels from PyPI or greyhound from source.
---

**greyhound** requires Python >= 3.11, PyTorch >= 2.7.1, and a CUDA-capable GPU.

## Installing with `pip`

**greyhound-kernels** is available [on PyPI](https://pypi.org/project/greyhound-kernels/).
It provides the `greyhound` Python import package:

```bash
pip install greyhound-kernels
```

Or with `uv`:

```bash
uv add greyhound-kernels
```

## Installing from source

Clone [the repository](https://github.com/tyler-romero/greyhound):

```bash
git clone https://github.com/tyler-romero/greyhound.git
cd greyhound
```

Then install in editable mode with development dependencies:

### uv

```bash
uv sync --all-extras
```

### pip

```bash
pip install -e ".[dev]"
```

## Optional dependencies

For benchmarking against third-party kernel libraries (Linux only):

### uv

```bash
uv sync --extra thirdparty
```

### pip

```bash
pip install -e ".[thirdparty]"
```

This installs [Dion](https://github.com/microsoft/dion),
[liger-kernel](https://github.com/linkedin/liger-kernel),
[quack-kernels](https://github.com/Dao-AILab/quack),
[gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for
comparison benchmarks.

For only the `causal-conv1d` comparison provider:

### uv

```bash
uv sync --extra causal-conv1d
```

### pip

```bash
pip install -e ".[causal-conv1d]"
```

For only the Quack dependency used by bonus Newton-Schulz utilities:

### uv

```bash
uv sync --extra quack
```

### pip

```bash
pip install -e ".[quack]"
```

For Modal benchmark execution:

### uv

```bash
uv sync --extra modal
```

### pip

```bash
pip install -e ".[modal]"
```
