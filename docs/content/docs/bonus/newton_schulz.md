---
title: Newton-Schulz
description: Bonus Muon-style Newton-Schulz orthogonalization using provider kernels.
---

Newton-Schulz orthogonalization is the matrix iteration used by Muon-style
optimizers to turn a gradient matrix into an approximate zeroth power. Greyhound exposes
this as a **bonus operation**, since it leverages [Quack's](https://github.com/Dao-AILab/quack) [symmetric GEMM](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_symmetric.py) and not any CuTe kernel
implemented in Greyhound. This exact implementation was suggested in [this blog post](https://dao-lab.ai/blog/2026/gram-newton-schulz/#kernel-optimizations-for-standard-newton-schulz) by Zhang et al.

```python
from greyhound.bonus.newton_schultz import orthogonalize_via_newton_schulz
```

## Iteration

Greyhound performs orthogonalization using the quintic Newton-Schulz iteration:

$$X_{k+1} = aX_k + (bA_k + cA_k^2)X_k,\qquad A_k = X_kX_k^T$$

Here, `A_k` is a symmetric Gram matrix formed from the current iterate. Symmetric GEMM is used to efficiently compute `A_k` and `bA_k + cA_k^2`. This optimization was first proposed in
[Faster Symmetric Matrix Multiplication with ThunderKittens](https://www.lakernewhouse.com/assets/writing/faster-symmul-with-thunderkittens.pdf).

The input is first cast to bfloat16 and normalized by its Frobenius norm. Tall matrices
are transposed internally so the symmetric products operate on the smaller matrix
dimension, then the result is transposed back to the original shape.

Each iteration composes:

1. Quack symmetric GEMM for `A = X @ X.T`
2. Quack symmetric GEMM for `bA + c(A @ A)`
3. PyTorch `addmm` / `baddbmm` for `aX + update @ X`

By default, the function uses the five-step quintic coefficient schedule in
`DEFAULT_NS_CONSTS`. Callers can pass a custom sequence of `(a, b, c)` constants with
`ns_consts`.

## Shape and dtype behavior

`orthogonalize_via_newton_schulz` accepts a single matrix `[M, N]` or a batched tensor
`[B, M, N]`. It runs the iteration in bfloat16 and returns a tensor with the same dtype
and shape as the input.


## Provider requirements

At runtime, `orthogonalize_via_newton_schulz` requires Quack symmetric GEMM support on
the current CUDA device. Quack's symmetric GEMM path currently requires CUDA and an
SM90-or-newer NVIDIA GPU.

Install the Quack extra:

```bash
pip install "greyhound-kernels[quack]"
```

## Usage

```python
import torch
from greyhound.bonus.newton_schultz import orthogonalize_via_newton_schulz

g = torch.randn(1024, 3072, device="cuda", dtype=torch.bfloat16)
out = orthogonalize_via_newton_schulz(g)

batched = torch.randn(8, 1024, 3072, device="cuda", dtype=torch.bfloat16)
out = orthogonalize_via_newton_schulz(batched)

ns_consts = [
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
]
out = orthogonalize_via_newton_schulz(g, ns_consts=ns_consts)
```

## Benchmarks

Benchmarks use bfloat16 inputs, five Newton-Schulz iterations, and representative Qwen3
matrix shapes. The interactive plot uses `m` and `n` as the 3D axes and provides a
dropdown selector for `batch_size`. The z-axis is median runtime in milliseconds, so
lower values are faster.

The benchmark compares:

- `greyhound`: Quack symmetric GEMM plus PyTorch GEMM update
- `dion`: Dion's Triton Newton-Schulz implementation
- `gram-newton-schulz`: Dao-AILab's Gram Newton-Schulz implementation
- `torch-eager`: PyTorch baseline
- `torch-compile`: the same PyTorch baseline through `torch.compile`

<div class="gpu-benchmark-picker" data-gpu-picker>
  <label class="gpu-picker-control">
    <span>GPU</span>
    <select data-gpu-select aria-label="Benchmark GPU"></select>
  </label>

  <div data-gpu-panel data-gpu="NVIDIA GeForce RTX 4090">

<iframe class="plot-frame" title="Newton-Schulz forward speed" src="../../assets/plots_html/newton_schulz_forward_speed_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA H100 80GB HBM3" data-gpu-hidden>

<iframe class="plot-frame" title="Newton-Schulz forward speed on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/newton_schulz_forward_speed_dtype=bfloat16_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA B200" data-gpu-hidden>

<iframe class="plot-frame" title="Newton-Schulz forward speed on NVIDIA B200" src="../../assets/plots_html/newton_schulz_forward_speed_dtype=bfloat16_NVIDIA_B200.html" scrolling="no"></iframe>

  </div>
</div>

<script type="module" src="../../assets/gpu-picker.js"></script>
