# Greyhound

High-performance fused GPU kernels for PyTorch training workloads. Greyhound
implements standalone CuTe DSL kernels, wraps them as PyTorch custom ops, and
exposes them through `greyhound.nn.functional` plus a small set of `nn.Module`
wrappers.

> **Status**: Alpha (`0.1.x`). APIs and performance tradeoffs may change between
> releases.

## What Is Included

| Operation | Functional API | Module API | Notes |
| --- | --- | --- | --- |
| Cross-entropy | `greyhound.nn.functional.cross_entropy` | - | Standalone cross-entropy loss backed by a loss-and-logits-gradient kernel. |
| Chunked linear loss | `greyhound.nn.functional.chunked_linear_loss` | - | Chunks the final projection for user-provided losses; `chunked_linear_cross_entropy` adds a fused cross-entropy-and-logits-gradient kernel per logits tile. |
| Causal Conv1D | `greyhound.nn.functional.causal_conv1d` | `GreyhoundCausalConv1d` | Depthwise causal short convolution, optional SiLU. |
| Selective log-softmax | `greyhound.nn.functional.selective_log_softmax` | - | Gathers one log-probability per row without materializing full log-probs. |

Operations are built in three layers:

1. `greyhound.kernels`: raw CuTe DSL kernels.
2. `greyhound.ops`: `torch.library.custom_op` and autograd integration.
3. `greyhound.nn` / `greyhound.nn.functional`: user-facing modules and functions.

## Bonus Integrations

`greyhound.bonus` contains experimental utilities that compose
kernels from other providers. These are useful for trying higher-level training
building blocks without treating them as core Greyhound kernels.

| Utility | API | Provider dependency | Notes |
| --- | --- | --- | --- |
| Newton-Schulz orthogonalization | `greyhound.bonus.newton_schultz.orthogonalize_via_newton_schulz` | `quack-kernels` | Muon-style zeropower iteration using Quack symmetric GEMM when available. |

## Requirements

- Python 3.11+
- PyTorch 2.7.1+
- CUDA-capable NVIDIA GPU
- `nvidia-cutlass-dsl[cu13]` on non-macOS platforms

Some benchmarks and bonus paths compare against optional third-party providers:
`liger-kernel`, `quack-kernels`, `flash-linear-attention`, `causal-conv1d`,
`gram-newton-schulz`, and `dion`.

## Installation

From PyPI:

```bash
pip install greyhound-kernels
```

From source:

```bash
git clone https://github.com/tyler-romero/greyhound.git
cd greyhound
pip install -e ".[dev]"
```

With the Quack bonus and benchmark provider:

```bash
pip install -e ".[dev,quack]"
```

With the full local benchmark provider set:

```bash
uv sync --extra dev --group thirdparty
```

With Modal benchmark tooling:

```bash
pip install -e ".[dev,modal]"
```

## Quick Start

### Functional Ops

```python
import torch
from greyhound.nn.functional import (
    autograd_loss_and_logits_grad,
    causal_conv1d,
    cross_entropy,
    chunked_linear_cross_entropy,
    chunked_linear_loss,
    selective_log_softmax,
)

device = "cuda"

hidden = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
lm_head = torch.randn(128256, 4096, device=device, dtype=torch.bfloat16)
targets = torch.randint(0, 128256, (4096,), device=device)
loss = chunked_linear_cross_entropy(hidden, lm_head, targets)

regression_target = torch.randn(4096, device=device, dtype=torch.float32)

def mse_chunk(logits, target):
    prediction = logits.float().mean(dim=-1)
    return torch.nn.functional.mse_loss(
        prediction, target, reduction="sum"
    )

mse_loss_and_grad = autograd_loss_and_logits_grad(mse_chunk)
custom_loss = chunked_linear_loss(hidden, lm_head, mse_loss_and_grad, regression_target)

logits = torch.randn(16, 128256, device=device, dtype=torch.bfloat16)
labels = torch.randint(0, 128256, (16,), device=device)
ce_loss = cross_entropy(logits, labels)

index = torch.randint(0, 128256, (16,), device=device)
selected = selective_log_softmax(logits, index)

conv_x = torch.randn(4, 1024, 2048, device=device, dtype=torch.bfloat16)
conv_weight = torch.randn(1024, 4, device=device, dtype=torch.bfloat16)
conv_out = causal_conv1d(conv_x, conv_weight, activation="silu")
```

### Module Wrappers

```python
import torch
from greyhound.nn import GreyhoundCausalConv1d

device = "cuda"

conv = GreyhoundCausalConv1d(
    channels=1024,
    kernel_size=4,
    activation="silu",
    device=device,
    dtype=torch.bfloat16,
)
```

## Benchmarks

Benchmark entry points live in `src/benchmarks`:

- `causal_conv1d_bench.py`
- `cross_entropy_with_grad_bench.py`
- `chunked_linear_cross_entropy_bench.py`
- `logprobs_bench.py`
- `newton_schulz_bench.py`

Run one locally:

```bash
uv run python src/benchmarks/chunked_linear_cross_entropy_bench.py --mode full
```

Run the default benchmark set on Modal and merge results back into the local CSV:

```bash
uv run --extra modal python scripts/run_modal_benchmarks.py --gpu H100
```

Regenerate plots from CSV data:

```bash
uv run python src/benchmarks/plot_from_csv.py
```

## Development

Use `uv` for local development:

```bash
uv sync --extra dev
```

Common checks:

```bash
ruff format .
ruff check .
ty check .
pytest -v --color=yes --doctest-modules src/tests/ src/greyhound/
```

The project is intentionally typed as a Python package (`py.typed`). CuTe DSL
kernel files are excluded from `ty` because their DSL constructs do not map cleanly
to Python static typing.

## Documentation

The docs site includes installation notes, API docs, per-kernel pages, and
benchmark visualizations:

https://tyler-romero.github.io/greyhound/

Build docs locally:

```bash
make docs-build
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
