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

For the Quack dependency used by bonus Newton-Schulz utilities and Quack
comparison benchmarks:

### uv

```bash
uv sync --extra quack
```

### pip

```bash
pip install -e ".[quack]"
```

For the full benchmark provider set:

```bash
uv sync --extra dev --group thirdparty
```

This uv-only dependency group installs comparison providers used by benchmark scripts, such as
[Dion](https://github.com/microsoft/dion),
[liger-kernel](https://github.com/linkedin/liger-kernel),
[gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and
`causal-conv1d`. The group is intentionally not published as a Greyhound package
extra because PyPI rejects direct Git dependencies in uploaded package metadata.

For Modal benchmark execution:

### uv

```bash
uv sync --extra modal
```

### pip

```bash
pip install -e ".[modal]"
```
