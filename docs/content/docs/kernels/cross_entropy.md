---
title: Cross-Entropy
description: Cross-entropy loss and logits gradient in one streaming pass.
---

Greyhound's standalone cross-entropy kernel computes the summed loss and the
gradient with respect to logits in the same pass over the logits tensor.

This is the lower-level kernel used by Chunked Linear Loss's cross-entropy
specialization after each logits tile is produced. Unlike Chunked Linear Loss, it assumes the
logits already exist as a `[batch, vocab]` tensor. The memory win is therefore
smaller than chunking the final projection, but the kernel still avoids a separate
softmax/log-softmax allocation and writes `dlogits` back into the logits buffer
in-place when called through the raw helper.

For each row, it computes:

$$\text{loss}_i = \log \sum_j \exp(\text{logits}_{i,j}) - \text{logits}_{i,y_i}$$

When z-loss is enabled, the kernel also accumulates:

$$z_i = \left(\log \sum_j \exp(\text{logits}_{i,j})\right)^2$$

## Kernel design

### Loss and gradient together

The kernel launches one block per row. Each block streams across the vocabulary
dimension and computes a numerically stable log-sum-exp with fp32 online
accumulation:

1. Each thread scans a strided slice of vocabulary columns.
2. Warp reductions combine local max/sum pairs.
3. A final warp reduction produces the row-level log-sum-exp.
4. Thread 0 atomically adds the row loss, optional z-loss term, and valid-token
   count to fp32 scalar accumulators.
5. All threads make a second pass over their columns and overwrite logits with
   `dlogits`.

The gradient written back to logits is the usual softmax-minus-one-hot term. With
z-loss enabled, the softmax term is scaled by the z-loss derivative before the
target column is adjusted.

### Shape and dtype constraints

`logits` must be contiguous with shape `[batch, vocab]`, and `labels` must be a
contiguous int64 tensor with shape `[batch]`. The kernel supports `float16`,
`bfloat16`, and `float32` logits.

Rows whose label equals `ignore_index` do not contribute to the loss or valid-token
count, and their logits gradient is written as zero.

### Relationship to Chunked Linear Loss

Use `chunked_linear_cross_entropy` when model code computes a final projection
immediately before cross-entropy. It avoids materializing full logits by chunking
the projection and calling this cross-entropy kernel on each temporary tile.

Use this lower-level kernel when logits already exist and you specifically want
the loss and `dlogits` in one kernel. The functional wrapper exposes normal
autograd semantics without mutating the input logits; the raw kernel helper
overwrites its logits input in-place.

## Usage

### Functional API

```python
import torch
from greyhound.nn.functional import cross_entropy

logits = torch.randn(8192, 128256, device="cuda", dtype=torch.bfloat16)
target = torch.randint(0, logits.shape[-1], (logits.shape[0],), device="cuda")

loss = cross_entropy(logits, target)
loss.backward()
```

`greyhound.nn.functional.cross_entropy`

### Raw kernel helper

```python
import torch
from greyhound.kernels.cross_entropy import cross_entropy_with_grad_kernel

logits = torch.randn(8192, 128256, device="cuda", dtype=torch.bfloat16)
target = torch.randint(0, logits.shape[-1], (logits.shape[0],), device="cuda")

# The raw kernel overwrites logits in-place with dlogits.
ce_sum, z_sum, n_valid = cross_entropy_with_grad_kernel(
    logits,
    target,
    z_loss_multiplier=0.0,
    ignore_index=-100,
)

loss = ce_sum / n_valid.clamp(min=1.0)
dlogits = logits
```

`greyhound.kernels.cross_entropy.cross_entropy_with_grad_kernel`

## Benchmarks

Benchmarks use bfloat16 logits, `reduction="sum"`, and no z-loss. The x-axis is
batch size and the y-axis is vocabulary size. The operation mode is listed as
forward because Greyhound and Liger both compute and store `dlogits` during the
forward kernel, so there is no separate backward kernel to time.

<div class="gpu-benchmark-picker" data-gpu-picker>
  <label class="gpu-picker-control">
    <span>GPU</span>
    <select data-gpu-select aria-label="Benchmark GPU"></select>
  </label>

  <div data-gpu-panel data-gpu="NVIDIA GeForce RTX 4090">

<iframe class="plot-frame" title="Cross-entropy forward speed" src="../../assets/plots_html/cross_entropy_forward_speed_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Cross-entropy forward memory usage" src="../../assets/plots_html/cross_entropy_forward_memory_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA H100 80GB HBM3" data-gpu-hidden>

<iframe class="plot-frame" title="Cross-entropy forward speed on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/cross_entropy_forward_speed_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Cross-entropy forward memory usage on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/cross_entropy_forward_memory_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA B200" data-gpu-hidden>

<iframe class="plot-frame" title="Cross-entropy forward speed on NVIDIA B200" src="../../assets/plots_html/cross_entropy_forward_speed_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_B200.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Cross-entropy forward memory usage on NVIDIA B200" src="../../assets/plots_html/cross_entropy_forward_memory_dtype=bfloat16_reduction=sum_z_loss_multiplier=0.0_NVIDIA_B200.html" scrolling="no"></iframe>

  </div>
</div>

<script type="module" src="../../assets/gpu-picker.js"></script>
