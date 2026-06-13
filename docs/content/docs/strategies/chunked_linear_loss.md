---
title: Chunked Linear Loss
description: Strategy notes for chunking a final linear projection before applying a scalar loss.
---

Chunked Linear Loss is a scheduling strategy for the common pattern:

```python
logits = inputs @ weight.T
loss = loss_fn(logits, ...)
```

Instead of materializing the full `[batch, vocab]` or `[tokens, vocab]` logits
tensor, Greyhound computes the projection in row chunks, applies the loss to each
temporary logits tile, computes the tile's logits gradient, and immediately folds
that gradient back into `grad_input` and `grad_weight`.

The user-facing API is `greyhound.nn.functional.chunked_linear_loss`. The
cross-entropy specialization, `chunked_linear_cross_entropy`, uses the same
schedule but supplies a custom cross-entropy-and-logits-gradient kernel for each
temporary tile.

## Why chunk the projection?

For language-model losses, the final projection often dominates activation
memory. If `inputs` is `[batch, hidden]` and `weight` is `[vocab, hidden]`, the
ordinary projection produces:

```python
logits = inputs @ weight.T  # [batch, vocab]
```

That tensor can be much larger than the inputs or model weights. For example,
8192 tokens and a 128k vocabulary produce more than one billion logits. In bf16,
that is roughly 2 GiB for logits alone before accounting for softmax workspaces,
autograd state, gradients, allocator padding, and other live tensors.

Most losses do not need all rows of that tensor at once. Cross-entropy, MSE over
rows, smooth L1, binary cross-entropy, and many custom scalar losses can be
evaluated independently over row slices as long as their reductions are additive.
Chunked Linear Loss exploits that property by changing the lifetime of logits:

1. Compute `logits_chunk = inputs[start:end] @ weight.T`.
2. Call a user-provided function that returns both a scalar loss and `dlogits_chunk`.
4. Accumulate `grad_input[start:end] = dlogits_chunk @ weight`.
5. Accumulate `grad_weight += dlogits_chunk.T @ inputs[start:end]`.
6. Discard the logits tile and move to the next chunk.

The full logits tensor never exists.

## What is fused?

The generic `chunked_linear_loss` path is primarily a chunked schedule, not a
single monolithic fused kernel. The projection uses matmul, the user-provided
callback supplies the loss and logits gradient for each chunk, and Greyhound
performs chunked input-gradient and weight-gradient accumulation around that
callback.

The optimized `chunked_linear_cross_entropy` specialization is fused in a narrower
sense: for each logits tile, the cross-entropy loss and logits gradient are
computed together by a single kernel that overwrites the temporary logits tile
with `dlogits`. The end-to-end operation is still chunked over rows; the
cross-entropy work inside each tile is fused.

## Loss function contract

`chunked_linear_loss` accepts a Python callable that returns both the chunk loss
and the gradient of that loss with respect to the logits chunk:

```python
loss = chunked_linear_loss(inputs, weight, loss_and_grad_fn, *loss_args, **loss_kwargs)
```

For each chunk, Greyhound calls:

```python
loss, grad_logits = loss_and_grad_fn(logits_chunk, *chunk_args, **chunk_kwargs)
```

The loss must be a scalar tensor. `grad_logits` must have the same shape as
`logits_chunk`. The function does not need to know the chunk start or end
indices; Greyhound handles slicing any per-row arguments.

The explicit `grad_logits` return is the core strategy boundary. It lets optimized
loss implementations compute the loss and gradient together, while keeping the
chunked projection and weight-gradient accumulation generic.

## Adapting PyTorch losses

Most built-in PyTorch losses return only a scalar loss. Use
`autograd_loss_and_logits_grad` to adapt them:

```python
from greyhound.nn.functional import autograd_loss_and_logits_grad, chunked_linear_loss

loss = chunked_linear_loss(
    inputs,
    weight,
    autograd_loss_and_logits_grad(torch.nn.functional.cross_entropy),
    target,
    reduction="sum",
)
```

The adapter runs PyTorch autograd on the temporary logits tile:

```python
with torch.enable_grad():
    logits_for_grad = logits_chunk.detach().requires_grad_(True)
    loss = loss_fn(logits_for_grad, ...)
    grad_logits = torch.autograd.grad(loss, logits_for_grad)
```

This is the generic path for ordinary differentiable PyTorch losses. It is
convenient, but it is not as specialized as a callback that computes
`grad_logits` directly.

## Direct logits gradients

For maximum control, provide a callback that computes the loss and logits gradient
itself:

```python
def weighted_mse_loss_and_grad(logits, target, row_weight, feature_weight):
    error = logits.float() - target
    weights = row_weight[:, None] * feature_weight[None, :]
    loss = (error.square() * weights).sum()
    grad_logits = (2 * error * weights).to(logits.dtype)
    return loss, grad_logits

loss = chunked_linear_loss(
    inputs,
    weight,
    weighted_mse_loss_and_grad,
    regression_target,
    row_weight,
    feature_weight,
)
```

This is the same boundary used by optimized specializations such as
`chunked_linear_cross_entropy`: the loss implementation owns the math for
`dlogits`; the chunked-linear strategy owns projection chunking and accumulation
back into `inputs` and `weight`.

## Sliced arguments

Tensor positional and keyword arguments whose leading dimension equals
`inputs.shape[0]` are sliced to the current chunk. Other values are passed through
unchanged.

This means per-row targets are sliced:

```python
target.shape == (batch,)
target_chunk = target[start:end]
```

Per-row regression targets are also sliced:

```python
target.shape == (batch, features)
target_chunk = target[start:end]
```

Global tensors are not sliced if their leading dimension does not match the batch
dimension. For example, a per-class `pos_weight` of shape `[vocab]` for binary
cross-entropy is passed through unchanged.

This rule keeps the API simple, but it is intentionally conservative. If a tensor
has leading dimension equal to `inputs.shape[0]`, Greyhound treats it as row-wise
data. If that is not what you want, wrap your loss in a small adapter that closes
over the tensor instead of passing it as a sliced argument.

## Reduction semantics

Chunked losses compose cleanly when the per-chunk loss values are additive.
Prefer `reduction="sum"` for existing PyTorch losses wrapped with
`autograd_loss_and_logits_grad`:

```python
loss = chunked_linear_loss(
    inputs,
    weight,
    autograd_loss_and_logits_grad(F.cross_entropy),
    target,
    reduction="sum",
)
```

Using `reduction="mean"` inside each chunk usually changes the result, because it
computes an average per chunk and then sums those chunk averages. That is not the
same as averaging over the full batch unless all chunks have identical effective
sizes and masks.

If you need a true global mean for a custom loss, return a summed loss from the
chunk function and divide outside by the global denominator you want.

`chunked_linear_cross_entropy` handles `"sum"` and `"mean"` itself because it
tracks the total valid-token count across chunks.

## Gradient handling

For the generic path, Greyhound delegates logits-gradient computation to the
callback:

```python
loss_chunk, grad_logits = loss_and_grad_fn(logits_chunk, ...)
```

Then Greyhound accumulates:

```python
grad_input[start:end] = grad_logits @ weight
grad_weight += inputs[start:end].T @ grad_logits
```

The later outer autograd backward pass returns the precomputed `grad_input` and
`grad_weight`, scaled by the upstream scalar gradient. This is unusual compared
with a normal PyTorch graph, but it is the part that avoids storing or recomputing
the full logits activation. The `autograd_loss_and_logits_grad` helper implements
the callback by running PyTorch autograd per chunk; optimized callbacks can return
`grad_logits` without invoking autograd.

## Weight-gradient accumulation dtype

`grad_weight_accum_dtype` controls the dtype used while accumulating the
chunk-by-chunk weight gradient:

- `"fp32"` accumulates partial `inputs.T @ grad_logits` products in fp32, then
  casts once to the parameter dtype. This is the default because it matches the
  usual mixed-precision training pattern.
- `"weight"` accumulates directly in the weight dtype. This can reduce memory and
  improve speed, but it rounds between chunks.

The matmuls still run in the input and weight dtypes. The option only controls
the accumulator used to combine per-chunk weight-gradient contributions.

## Chunk size

When `chunk_size` is omitted, Greyhound chooses a memory-aware row chunk size from
the batch size, hidden dimension, vocabulary size, and element size. The heuristic
tries to keep temporary logits tiles large enough for efficient matmul while
bounded enough to avoid materializing a full `[batch, vocab]` tensor.

You can pass `chunk_size` explicitly when you need predictable memory behavior or
want to tune a specific model shape:

```python
loss = chunked_linear_loss(inputs, weight, loss_and_grad_fn, target, chunk_size=1024)
```

Smaller chunks reduce peak logits memory, but increase loop overhead and may make
the projection matmuls less efficient. Larger chunks improve matmul efficiency but
increase peak memory.

## When to use it

Use `chunked_linear_loss` when:

- The final projection output is large enough that full logits are expensive.
- Your loss can be expressed as a sum of per-row or per-token contributions.
- You want to bring an existing PyTorch loss through `autograd_loss_and_logits_grad`.
- You have a custom loss implementation that can return `grad_logits` directly.

Use `chunked_linear_cross_entropy` when:

- The loss is ordinary class-index cross-entropy.
- You want the optimized tile-level cross-entropy-and-logits-gradient kernel.
- You need correct global `"mean"` semantics with `ignore_index`.
- You want optional z-loss support.

Do not use the generic strategy for losses that need to compare all rows at once,
such as batch-level contrastive losses, in-batch negatives, or losses whose
normalization depends on the full logits tensor across the chunked dimension.

## Relationship to `torch.compile`

`torch.compile` can optimize a program that already materializes `logits`, but it
cannot generally transform:

```python
loss = loss_fn(inputs @ weight.T, target)
```

into a schedule that changes the lifetime of the intermediate `[batch, vocab]`
tensor. Chunked Linear Loss is useful because it changes the program structure:
the logits tile is produced, consumed, converted into gradients, and discarded
before the next tile is computed.
