---
title: greyhound
description: High-performance GPU kernels written in CuTe DSL.
hero:
  tagline: High-performance GPU kernels written in CuTe DSL.
  image:
    html: '<img src="assets/logo.png" alt="Greyhound logo" width="240" height="240" loading="eager" decoding="async">'
  actions:
    - text: Get Started
      link: installation/
      icon: right-arrow
    - text: View Kernels
      link: "#kernels"
---

Greyhound is a Python library for high-performance, GPU kernels targeting
PyTorch training and inference workloads. It implements standalone CuTe DSL kernels for common
transformer operations, then wraps them with PyTorch custom ops so they integrate
with `torch.compile`.

```bash
pip install greyhound-kernels
```

```python
from greyhound.nn import GreyhoundCausalConv1d
from greyhound.nn.functional import (
    causal_conv1d,
    chunked_linear_cross_entropy,
    selective_log_softmax,
)
```

## How Greyhound Fits Together

Each kernel is exposed through three layers, so you can choose the right level for
your model code:

### Raw kernels

`greyhound.kernels` contains the low-level standalone CuTe DSL kernels that define
the GPU computation.

### Ops and functional API

`greyhound.ops` and `greyhound.nn.functional` wrap the raw kernels with
`@torch.library.custom_op` for `torch.compile` compatibility. Operations include
fake tensor registrations for symbolic tracing, autograd bridges where needed, and
public functions such as `chunked_linear_cross_entropy()`, `causal_conv1d()`, or
`selective_log_softmax()`.

This is the recommended layer for most users.

### Module wrappers

`greyhound.nn` provides standard PyTorch `nn.Module` classes that call the
functional layer. These are drop-in replacements for common model blocks:

- `GreyhoundCausalConv1d` replaces depthwise causal `Conv1d` layers

## Kernels

Greyhound provides fused GPU kernels for operations commonly found in
transformer-based models. Each kernel page covers the kernel design, fusion
or scheduling strategy, usage examples, and benchmark results.

### [Chunked Linear Cross-Entropy](kernels/chunked_linear_cross_entropy/)

Uses a chunked final projection and a fused cross-entropy-and-logits-gradient
kernel per logits tile, avoiding materialization of the full `[tokens, vocab]`
logits tensor.

Functional API: `greyhound.nn.functional.chunked_linear_cross_entropy`

### [Cross-Entropy](kernels/cross_entropy/)

Computes cross-entropy loss with a standalone loss-and-logits-gradient kernel.
The raw helper overwrites logits with `dlogits`; the functional wrapper preserves
the input tensor and exposes normal autograd semantics.

Functional API: `greyhound.nn.functional.cross_entropy`

Raw kernel helper: `greyhound.kernels.cross_entropy.cross_entropy_with_grad_kernel`

### [Causal Conv1D](kernels/causal_conv1d/)

Runs depthwise causal convolution with masked loads for state-space and sequence
model blocks.

Module wrapper: `GreyhoundCausalConv1d`

Functional API: `greyhound.nn.functional.causal_conv1d`

### [Selective Log Softmax](kernels/selective_log_softmax/)

Streams the vocabulary dimension to gather one log-probability per row without
materializing full log-probabilities.

Functional API: `greyhound.nn.functional.selective_log_softmax`

## Strategies

Strategy pages cover higher-level scheduling patterns that may compose kernels,
PyTorch autograd, and custom ops.

### [Chunked Linear Loss](strategies/chunked_linear_loss/)

Explains the memory-lifetime strategy behind `chunked_linear_loss`, including
sliced loss arguments, reduction semantics, gradient accumulation, chunk sizing,
and when to use the optimized cross-entropy specialization.

## Bonus Integrations

`greyhound.bonus` contains experimental utilities that compose Greyhound code with
kernels from other providers. These are not core CuTe DSL kernels, but they are useful
for trying higher-level training building blocks.

### [Newton-Schulz](bonus/newton_schulz/)

Runs Muon-style Newton-Schulz orthogonalization using Quack symmetric GEMM.

Functional API: `greyhound.bonus.newton_schultz.orthogonalize_via_newton_schulz`

## Quick Shape

```python
import torch
from greyhound.nn import GreyhoundCausalConv1d
from greyhound.nn.functional import chunked_linear_cross_entropy

x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.bfloat16)
conv_x = torch.randn(8, 4096, 2048, device="cuda", dtype=torch.bfloat16)
lm_head = torch.randn(128256, 4096, device="cuda", dtype=torch.bfloat16)
targets = torch.randint(0, 128256, (8 * 2048,), device="cuda")

conv = GreyhoundCausalConv1d(4096, kernel_size=4, activation="silu").cuda()

loss = chunked_linear_cross_entropy(x.reshape(-1, 4096), lm_head, targets)
z = conv(conv_x)
```
