---
title: Module Wrappers
description: Generated reference for greyhound.nn module wrappers.
---

Generated from Python docstrings by `scripts/generate_api_docs.py`.

### `GreyhoundCausalConv1d`

`greyhound.nn.causal_conv1d.GreyhoundCausalConv1d`

```python
GreyhoundCausalConv1d(channels: int, kernel_size: int, bias: bool = True, activation: str | None = None, device: torch.device | str | None = None, dtype: torch.dtype | None = None)
```

Causal 1D depthwise convolution with optional SiLU activation.

Wraps the fused Greyhound kernel as a drop-in replacement for depthwise
`nn.Conv1d` with causal (left) padding. The kernel handles boundary
conditions internally via masked loads, avoiding the cost of explicit
`F.pad()`.

Inherits from `nn.Conv1d` with `groups=channels` (depthwise),
`stride=1`, `dilation=1`, and `padding=0` (causal padding is
applied inside the kernel).

**Parameters**
- `channels` (int): Number of input (and output) channels.
- `kernel_size` (int): Width of the convolution kernel (typically 2-4).
- `bias` (bool, default: `True`): If `True`, adds a learnable bias. Default: `True`.
- `activation` (str | None, default: `None`): Activation to apply after convolution. `"silu"` for
  SiLU activation, or `None` for no activation. Default: `None`.
- `device` (torch.device | str | None, default: `None`): Device for parameter initialization.
- `dtype` (torch.dtype | None, default: `None`): Data type for parameters.

#### `forward`

```python
forward(input: torch.Tensor) -> torch.Tensor
```

Apply causal depthwise convolution.

**Parameters**
- `input` (torch.Tensor): Input tensor of shape `[B, D, T]`.

**Returns**

`torch.Tensor`

Output tensor of shape `[B, D, T]`.
