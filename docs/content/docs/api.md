---
title: API Reference
description: Public functional operators and module wrappers exposed by greyhound.
---

Greyhound exposes fused kernels through functional operators and small `nn.Module` wrappers. The functional layer is the most direct way to call kernels from model code, while the module layer is useful when replacing standard transformer blocks.

Detailed API reference pages are generated from Python docstrings with `scripts/generate_api_docs.py`.

## Functional API

| Function | Purpose |
| --- | --- |
| `greyhound.nn.functional.cross_entropy` | Cross-entropy using the standalone loss-and-logits-gradient kernel. |
| `greyhound.nn.functional.autograd_loss_and_logits_grad` | Adapter from scalar PyTorch losses to chunked loss-and-logits-gradient callbacks. |
| `greyhound.nn.functional.chunked_linear_loss` | Chunked linear projection with a user-provided per-chunk loss-and-logits-gradient function. |
| `greyhound.nn.functional.chunked_linear_cross_entropy` | Chunked linear cross-entropy using a fused cross-entropy-and-logits-gradient kernel per logits tile. |
| `greyhound.nn.functional.causal_conv1d` | Depthwise causal 1D convolution with optional bias and activation. |
| `greyhound.nn.functional.selective_log_softmax` | Selected log-softmax values without materializing full log-probabilities. |

[Generated functional reference](functional/)

## nn.Module Wrappers

| Module | Purpose |
| --- | --- |
| `greyhound.nn.causal_conv1d.GreyhoundCausalConv1d` | Depthwise causal `Conv1d` wrapper backed by Greyhound's fused convolution kernel. |

[Generated module wrapper reference](modules/)

## Bonus Integrations

| Utility | Purpose |
| --- | --- |
| `greyhound.bonus.newton_schultz.orthogonalize_via_newton_schulz` | Muon-style Newton-Schulz orthogonalization composed with Quack symmetric GEMM when available. |

Bonus integrations are experimental utilities that compose Greyhound code with kernels
from other providers. They are documented separately from the core functional and
module API because they may require optional provider dependencies.

## Source Modules

The public API is implemented in the Python package under `src/greyhound/`:

| Surface | Source |
| --- | --- |
| Functional namespace | `src/greyhound/nn/functional.py` |
| Module wrappers | `src/greyhound/nn/` |
| Autograd ops | `src/greyhound/ops/` |
| CuTe DSL kernels | `src/greyhound/kernels/` |
| Bonus integrations | `src/greyhound/bonus/` |
