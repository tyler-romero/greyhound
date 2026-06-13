---
title: Causal Conv1D
description: Depthwise causal convolution with masked loads.
---

Causal 1D depthwise convolution with optional bias and SiLU activation. This operation is commonly used in state-space models (Mamba) and other sequence architectures where each position should only depend on current and past inputs.

It computes:

$$\text{out}[b, d, t] = \text{bias}[d] + \sum_{w=0}^{W-1} \text{weight}[d, w] \cdot x[b, d, t + w - (W-1)]$$

where $x[b, d, k] = 0$ for $k < 0$ (causal zero-padding), followed by optional SiLU activation.

## Kernel design

### Masked loads instead of padding

The most important design decision in this kernel is avoiding `F.pad()`. A naive implementation would pre-pad the input tensor before the convolution, which allocates new GPU memory and performs a full memcpy. For large tensors this can dominate kernel runtime.

Instead, the kernel takes the unpadded input directly and masks boundary conditions inside
the CuTe DSL kernel:

```python
if t + wi >= W - 1:
    x_val = x[b, d, t + wi - (W - 1)]
```

Positions where `t + w < W - 1` (i.e., the causal padding region) are masked to zero without any memory allocation. This approach eliminated ~320 MB of extra GPU memory allocation at `B=8, D=5120, T=2048` and made the kernel ~2x faster than the padded version.

### Forward kernel

The forward kernel tiles over `[B, D, T]` and uses a compile-time loop over `W`
(typically 2--4). For each output position, it accumulates the dot product of the weight
vector with the masked input window, optionally adds bias, and optionally applies SiLU
activation.

### Backward kernel

The backward kernel computes three outputs in a single launch:
- `dx`: Input gradient, computed by correlating the upstream gradient with the weight
- `dweight_partial`: Partial weight gradients per time-tile, reduced externally
- `dbias_partial`: Partial bias gradients per time-tile, reduced externally

When SiLU activation was applied in the forward pass, the backward kernel recomputes the pre-activation values to derive the SiLU gradient, avoiding the need to store them from the forward pass.

### Comparison with `torch.compile`

The memory savings from masked loads cannot be replicated by `torch.compile`, which would still call `F.pad()` to create the padded tensor. The compile-time unrolling of the small kernel width (W=2--4) and the fused activation are also difficult for inductor to match in a single kernel.

## Usage

### Module API

`GreyhoundCausalConv1d` is a drop-in depthwise `nn.Conv1d` wrapper for `[B, D, T]`
inputs. It inherits from `nn.Conv1d` with `groups=channels`, `stride=1`,
`dilation=1`, and `padding=0`; causal padding is handled inside the fused kernel.

```python
import torch
from greyhound.nn import GreyhoundCausalConv1d

B, D, T, W = 8, 5120, 2048, 4
x = torch.randn(B, D, T, device="cuda", dtype=torch.bfloat16)

conv = GreyhoundCausalConv1d(
    channels=D,
    kernel_size=W,
    bias=True,
    activation="silu",
    device="cuda",
    dtype=torch.bfloat16,
)

out = conv(x)  # shape: [B, D, T]
```

`greyhound.nn.causal_conv1d.GreyhoundCausalConv1d`

### Functional API

```python
import torch
from greyhound.nn.functional import causal_conv1d

B, D, T, W = 8, 5120, 2048, 4
x = torch.randn(B, D, T, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(D, device="cuda", dtype=torch.bfloat16)

# Without activation
out = causal_conv1d(x, weight, bias)  # shape: [B, D, T]

# With SiLU activation
out = causal_conv1d(x, weight, bias, activation="silu")

# Without bias
out = causal_conv1d(x, weight)
```

`greyhound.nn.functional.causal_conv1d`

## Benchmarks

Benchmarks use bfloat16, batch=8, W=4, and activation=silu. The x-axis is the sequence length.

<div class="gpu-benchmark-picker" data-gpu-picker>
  <label class="gpu-picker-control">
    <span>GPU</span>
    <select data-gpu-select aria-label="Benchmark GPU"></select>
  </label>

  <div data-gpu-panel data-gpu="NVIDIA GeForce RTX 4090">

<iframe class="plot-frame" title="Causal Conv1D full speed" src="../../assets/plots_html/causal_conv1d_full_speed_activation=silu_batch_size=8_dtype=bfloat16_width=4_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Causal Conv1D forward speed" src="../../assets/plots_html/causal_conv1d_forward_speed_activation=silu_batch_size=8_dtype=bfloat16_width=4_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

<iframe class="plot-frame" title="Causal Conv1D backward speed" src="../../assets/plots_html/causal_conv1d_backward_speed_activation=silu_batch_size=8_dtype=bfloat16_width=4_NVIDIA_GeForce_RTX_4090.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA H100 80GB HBM3" data-gpu-hidden>

<iframe class="plot-frame" title="Causal Conv1D full speed on NVIDIA H100 80GB HBM3" src="../../assets/plots_html/causal_conv1d_full_speed_activation=silu_batch_size=8_dtype=bfloat16_width=4_NVIDIA_H100_80GB_HBM3.html" scrolling="no"></iframe>

  </div>

  <div data-gpu-panel data-gpu="NVIDIA B200" data-gpu-hidden>

<iframe class="plot-frame" title="Causal Conv1D full speed on NVIDIA B200" src="../../assets/plots_html/causal_conv1d_full_speed_activation=silu_batch_size=8_dtype=bfloat16_width=4_NVIDIA_B200.html" scrolling="no"></iframe>

  </div>
</div>

<script type="module" src="../../assets/gpu-picker.js"></script>
