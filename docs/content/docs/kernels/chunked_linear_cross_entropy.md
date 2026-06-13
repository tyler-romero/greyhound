---
title: Chunked Linear Cross-Entropy
description: Chunked linear cross-entropy using a fused cross-entropy-and-logits-gradient kernel per logits tile.
---

Chunked Linear Cross-Entropy is the optimized cross-entropy specialization of the
chunked linear loss strategy. It chunks the final projection, applies a fused
cross-entropy-and-logits-gradient kernel to each temporary logits tile, and avoids
materializing the full `[tokens, vocab]` logits tensor.

For the generic scheduling strategy and BYO loss API, see
[Strategies: Chunked Linear Loss](../../strategies/chunked_linear_loss/).

## Kernel design

The implementation chunks over the batch/token dimension. For each chunk,
it:

1. Computes a temporary logits tile: `logits_chunk = x_chunk @ W`
2. Runs the fused cross-entropy-and-logits-gradient kernel on that tile
3. Adds the chunk's loss into the total loss
4. Reuses the tile as `dlogits`, the gradient with respect to the logits
5. Immediately uses `dlogits` to accumulate the input and weight gradients

The standalone cross-entropy kernel computes the numerically stable row loss and
the logits gradient together. The outer autograd backward step scales and returns
the chunk-accumulated `grad_input` and `grad_weight`.

## Usage

### Functional API

```python
import torch
from greyhound.nn.functional import chunked_linear_cross_entropy

# inputs: [num_tokens, hidden_dim], weight: [vocab_size, hidden_dim]
inputs = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(32000, 4096, device="cuda", dtype=torch.bfloat16)
target = torch.randint(0, 32000, (8192,), device="cuda")

loss = chunked_linear_cross_entropy(inputs, weight, target)
loss.backward()

# With z-loss regularization
loss = chunked_linear_cross_entropy(inputs, weight, target, z_loss_multiplier=1e-4)

# With sum reduction and ignore_index
loss = chunked_linear_cross_entropy(
    inputs, weight, target,
    ignore_index=-100,
    reduction="sum",
)

# Opt into lower-memory/faster weight-dtype gradient accumulation
loss = chunked_linear_cross_entropy(
    inputs,
    weight,
    target,
    grad_weight_accum_dtype="weight",
)
```

`greyhound.nn.functional.chunked_linear_cross_entropy`

## Benchmarks

Benchmarks use bfloat16 inputs and a batch size of 8192 tokens. The x-axis is the vocabulary size.

<div class="gpu-benchmark-picker" data-gpu-picker>
  <label class="gpu-picker-control">
    <span>GPU</span>
    <select data-gpu-select aria-label="Benchmark GPU"></select>
  </label>

  <div data-gpu-panel data-gpu="NVIDIA GeForce RTX 4090">

<h3>Full pass</h3>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy full speed" src="../../assets/plots_html/chunked_linear_cross_entropy_full_speed_batch_size=8192_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy full memory usage" src="../../assets/plots_html/chunked_linear_cross_entropy_full_memory_batch_size=8192_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<h3>Forward pass</h3>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy forward speed" src="../../assets/plots_html/chunked_linear_cross_entropy_forward_speed_batch_size=8192_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy forward memory usage" src="../../assets/plots_html/chunked_linear_cross_entropy_forward_memory_batch_size=8192_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<p>These standalone cross-entropy plots provide the forward-only comparison point for the
loss computation itself:</p>

<iframe class="plot-frame" title="Cross-entropy forward speed" src="../../assets/plots_html/cross_entropy_forward_speed_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Cross-entropy forward memory usage" src="../../assets/plots_html/cross_entropy_forward_memory_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<h3>Backward pass</h3>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy backward speed" src="../../assets/plots_html/chunked_linear_cross_entropy_backward_speed_batch_size=8192_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy backward memory usage" src="../../assets/plots_html/chunked_linear_cross_entropy_backward_memory_batch_size=8192_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA H100 80GB HBM3" data-gpu-hidden>

<h3>Full pass</h3>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy full speed on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/chunked_linear_cross_entropy_full_speed_batch_size=8192_dtype=bfloat16_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy full memory usage on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/chunked_linear_cross_entropy_full_memory_batch_size=8192_dtype=bfloat16_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

<h3>Standalone cross-entropy</h3>

<iframe class="plot-frame" title="Cross-entropy forward speed on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/cross_entropy_forward_speed_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Cross-entropy forward memory usage on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/cross_entropy_forward_memory_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA B200" data-gpu-hidden>

<h3>Full pass</h3>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy full speed on NVIDIA B200" src="../../assets/plots_html/chunked_linear_cross_entropy_full_speed_batch_size=8192_dtype=bfloat16_NVIDIA_B200.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Chunked Linear Cross-Entropy full memory usage on NVIDIA B200" src="../../assets/plots_html/chunked_linear_cross_entropy_full_memory_batch_size=8192_dtype=bfloat16_NVIDIA_B200.html" scrolling="no"></iframe>

<h3>Standalone cross-entropy</h3>

<iframe class="plot-frame" title="Cross-entropy forward speed on NVIDIA B200" src="../../assets/plots_html/cross_entropy_forward_speed_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_B200.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Cross-entropy forward memory usage on NVIDIA B200" src="../../assets/plots_html/cross_entropy_forward_memory_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_B200.html" scrolling="no"></iframe>

  </div>
</div>

<script type="module" src="../../assets/gpu-picker.js"></script>
