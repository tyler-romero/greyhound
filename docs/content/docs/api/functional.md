---
title: Functional API
description: Generated reference for greyhound.nn.functional.
---

Generated from Python docstrings by `scripts/generate_api_docs.py`.

### `cross_entropy`

`greyhound.nn.functional.cross_entropy`

```python
cross_entropy(logits: Tensor, target: Tensor, ignore_index: int = -100, reduction: Literal['sum', 'mean'] = 'mean', z_loss_multiplier: float = 0.0) -> Tensor
```

Compute cross-entropy loss with optional z-loss regularization.

This function expects logits that have already been materialized. It uses
Greyhound's standalone cross-entropy kernel to compute the summed loss and
logits gradient in one streaming pass, then exposes normal autograd behavior
for `logits`. Unlike the raw kernel helper, this functional wrapper does
not mutate the input `logits` tensor.

**Parameters**
- `logits` (Tensor): Contiguous logits tensor of shape `[B, V]`.
- `target` (Tensor): Target labels of shape `[B]` containing class indices in
  `[0, V)`.
- `ignore_index` (int, default: `-100`): Target value to ignore when computing the loss. Tokens
  with this target value will not contribute to the loss or gradient.
  Default: `-100`.
- `reduction` (Literal['sum', 'mean'], default: `'mean'`): Specifies the reduction to apply to the output. One of:
  - `"mean"`: divide the summed loss by the number of valid tokens
  - `"sum"`: return the summed loss
  Default: `"mean"`.
- `z_loss_multiplier` (float, default: `0.0`): Coefficient for the auxiliary z-loss term, which
  penalizes large log-partition function values. Set to `0.0` to
  disable. Default: `0.0`.

**Returns**

`Tensor`

The combined cross-entropy loss plus z-loss scaled by
`z_loss_multiplier`.

### `autograd_loss_and_logits_grad`

`greyhound.nn.functional.autograd_loss_and_logits_grad`

```python
autograd_loss_and_logits_grad(loss_fn: Callable[..., Tensor]) -> Callable[..., tuple[Tensor, Tensor]]
```

Adapt a scalar PyTorch loss function to return `(loss, grad_logits)`.

`chunked_linear_loss` expects a callable that returns both a scalar loss and
the gradient of that scalar with respect to the logits chunk. This helper wraps
ordinary differentiable loss functions, such as `torch.nn.functional.cross_entropy`
or `torch.nn.functional.mse_loss`, by running PyTorch autograd on the logits
argument inside each chunk.

**Parameters**
- `loss_fn` (Callable[..., Tensor]): Callable receiving `(logits_chunk, *args, **kwargs)` and returning
  a scalar tensor differentiable with respect to `logits_chunk`.

**Returns**

`Callable[..., tuple[Tensor, Tensor]]`

Callable receiving the same arguments as `loss_fn` and returning
`(loss, grad_logits)`.

### `chunked_linear_loss`

`greyhound.nn.functional.chunked_linear_loss`

```python
chunked_linear_loss(inputs: Tensor, weight: Tensor, loss_and_grad_fn: Callable[..., tuple[Tensor, Tensor]], *loss_args: Any, chunk_size: int | None = None, grad_weight_accum_dtype: Literal['fp32', 'weight'] = 'fp32', **loss_kwargs: Any) -> Tensor
```

Compute a chunked linear projection followed by a user-provided loss and logits gradient.

This function evaluates `inputs @ weight.T` in row chunks. For each chunk,
it calls `loss_and_grad_fn(logits_chunk, *chunk_args, **chunk_kwargs)`.
Tensor arguments whose leading dimension matches `inputs.shape[0]` are
sliced to the same row range as the logits chunk; other arguments are passed
through unchanged. The loss-and-gradient function must return
`(loss, grad_logits)`, where `loss` is a scalar tensor and
`grad_logits` has the same shape as `logits_chunk`. Use
`autograd_loss_and_logits_grad` to adapt ordinary PyTorch scalar loss
functions to this contract.

Gradients for `inputs` and `weight` are accumulated chunk-by-chunk from
the per-chunk logits gradient, avoiding materialization of the full logits
tensor.

**Parameters**
- `inputs` (Tensor): Input tensor of shape `[B, D]`.
- `weight` (Tensor): Linear weight matrix of shape `[V, D]`.
- `loss_and_grad_fn` (Callable[..., tuple[Tensor, Tensor]]): Callable receiving
  `(logits_chunk, *chunk_args, **chunk_kwargs)` and returning
  `(loss, grad_logits)` for that chunk.
- `loss_args` (Any, default: `()`): Additional positional arguments for `loss_and_grad_fn`. Tensor
  arguments with leading dimension `B` are sliced per chunk.
- `chunk_size` (int | None, default: `None`): Optional number of rows per logits chunk. Defaults to an
  internal memory-aware heuristic.
- `grad_weight_accum_dtype` (Literal['fp32', 'weight'], default: `'fp32'`): Accumulator dtype for the chunked weight-gradient
  reduction. `"fp32"` accumulates partial `inputs.T @ grad_logits`
  products in fp32 and casts once before returning the gradient.
  `"weight"` accumulates directly in the parameter dtype.
  Default: `"fp32"`.
- `loss_kwargs` (Any, default: `{}`): Additional keyword arguments for `loss_and_grad_fn`. Tensor
  values with leading dimension `B` are sliced per chunk.

**Returns**

`Tensor`

Sum of the scalar losses returned by `loss_and_grad_fn` for each chunk.

### `chunked_linear_cross_entropy`

`greyhound.nn.functional.chunked_linear_cross_entropy`

```python
chunked_linear_cross_entropy(inputs: Tensor, weight: Tensor, target: Tensor, ignore_index: int = -100, reduction: Literal['sum', 'mean'] = 'mean', z_loss_multiplier: float = 0.0, grad_weight_accum_dtype: Literal['fp32', 'weight'] = 'fp32') -> Tensor
```

Compute chunked linear cross-entropy with optional z-loss regularization.

This is a specialization of `chunked_linear_loss` for cross-entropy. It uses
the same chunked linear-gradient accumulation and a fused
cross-entropy-and-logits-gradient kernel per logits tile, avoiding materialization
of the full logits tensor in memory.
During training, the weight gradient is accumulated in fp32 by default and cast once
at the end. That is the conservative mixed-precision choice: the matmuls run in
bf16/fp16, while the chunked reduction over tokens keeps fp32 accumulator state.
Set `grad_weight_accum_dtype="weight"` to accumulate directly in the weight dtype
for lower memory and higher speed when that numerical tradeoff is acceptable.

**Parameters**
- `inputs` (Tensor): Input tensor of shape [BT, D] where BT is the number of tokens and D is
  the hidden dimension.
- `weight` (Tensor): Weight matrix of shape [V, D] where V is the vocabulary size.
- `target` (Tensor): Target labels of shape [BT] containing class indices in [0, V).
- `ignore_index` (int, default: `-100`): Target value to ignore when computing the loss. Tokens with this
  target value will not contribute to the loss or gradient. Default: -100.
- `reduction` (Literal['sum', 'mean'], default: `'mean'`): Specifies the reduction to apply to the output. One of:
  - "mean": the sum of the output will be divided by the number of valid tokens
  - "sum": the output will be summed
  Default: "mean".
- `z_loss_multiplier` (float, default: `0.0`): Coefficient for the auxiliary z-loss term, which penalizes
  large log-partition function values (log-sum-exp of logits). Set to 0.0 to disable.
  Default: 0.0.
- `grad_weight_accum_dtype` (Literal['fp32', 'weight'], default: `'fp32'`): Accumulator dtype for the chunked weight-gradient reduction.
  `"fp32"` accumulates partial `x.T @ grad_logits` products in fp32 and casts
  once before returning the gradient. `"weight"` accumulates directly in the
  parameter dtype, which is faster and uses less memory but rounds between chunks.
  Default: `"fp32"`.

**Returns**

`Tensor`

The combined cross-entropy loss plus z-loss (scaled by z_loss_multiplier).

### `causal_conv1d`

`greyhound.nn.functional.causal_conv1d`

```python
causal_conv1d(x: Tensor, weight: Tensor, bias: Tensor | None = None, activation: str | None = None) -> Tensor
```

Causal 1D depthwise convolution with optional bias and SiLU activation.

Computes `out[b, d, t] = bias[d] + sum_{w=0}^{W-1} weight[d, w] * x[b, d, t - (W-1-w)]`
where `x` is zero for negative indices, followed by optional SiLU activation.

**Parameters**
- `x` (Tensor): Input tensor of shape `[B, D, T]`.
- `weight` (Tensor): Convolution weight of shape `[D, W]` where `W` is the kernel width
  (typically 2-4).
- `bias` (Tensor | None, default: `None`): Optional bias of shape `[D]`.
- `activation` (str | None, default: `None`): Activation function to apply. `"silu"` for SiLU activation,
  or `None` for no activation.

**Returns**

`Tensor`

Output tensor of shape `[B, D, T]`.

### `selective_log_softmax`

`greyhound.nn.functional.selective_log_softmax`

```python
selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor
```

Select log-softmax values at `index` without materializing full log-probabilities.

Computes the equivalent of:
`torch.gather(logits.log_softmax(dim=-1), -1, index.unsqueeze(-1)).squeeze(-1)`.
The CuTe kernel streams the final dimension with an online log-sum-exp reduction and
writes one selected log-probability for each row.

**Parameters**
- `logits` (Tensor): Input logits of shape `[..., vocab_size]`.
- `index` (Tensor): Indices of shape `[...]` selecting one vocabulary entry per row.

**Returns**

`Tensor`

Selected log-probabilities with shape `[...]`.
