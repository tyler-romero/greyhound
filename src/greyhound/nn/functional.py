from collections.abc import Callable
from typing import Any
from typing import Literal, cast

import torch
from torch import Tensor

from greyhound.ops.causal_conv1d import CausalConv1dFunction
from greyhound.ops.chunked_linear_cross_entropy import ChunkedLinearCrossEntropyFunction
from greyhound.ops.chunked_linear_loss import ChunkedLinearLossFunction
from greyhound.ops.cross_entropy import CrossEntropyFunction
from greyhound.ops.logprobs import SelectiveLogSoftmaxFunction


def _slice_chunk_arg(value: Any, start: int, end: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[start:end]
    return value


def _check_scalar_loss(loss: Tensor) -> None:
    if loss.ndim != 0:
        raise ValueError(f"chunked loss function must return a scalar tensor, got {loss.shape}")


def causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str | None = None,
) -> Tensor:
    """
    Causal 1D depthwise convolution with optional bias and SiLU activation.

    Computes ``out[b, d, t] = bias[d] + sum_{w=0}^{W-1} weight[d, w] * x[b, d, t - (W-1-w)]``
    where ``x`` is zero for negative indices, followed by optional SiLU activation.

    Args:
        x: Input tensor of shape ``[B, D, T]``.
        weight: Convolution weight of shape ``[D, W]`` where ``W`` is the kernel width
            (typically 2-4).
        bias: Optional bias of shape ``[D]``.
        activation: Activation function to apply. ``"silu"`` for SiLU activation,
            or ``None`` for no activation.

    Returns:
        Output tensor of shape ``[B, D, T]``.
    """
    activation_str = activation if activation is not None else "none"
    return cast(Tensor, CausalConv1dFunction.apply(x, weight, bias, activation_str))


def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    """
    Select log-softmax values at ``index`` without materializing full log-probabilities.

    Computes the equivalent of:
    ``torch.gather(logits.log_softmax(dim=-1), -1, index.unsqueeze(-1)).squeeze(-1)``.
    The CuTe kernel streams the final dimension with an online log-sum-exp reduction and
    writes one selected log-probability for each row.

    Args:
        logits: Input logits of shape ``[..., vocab_size]``.
        index: Indices of shape ``[...]`` selecting one vocabulary entry per row.

    Returns:
        Selected log-probabilities with shape ``[...]``.
    """
    return cast(Tensor, SelectiveLogSoftmaxFunction.apply(logits, index))


def cross_entropy(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: Literal["sum", "mean"] = "mean",
    z_loss_multiplier: float = 0.0,
) -> Tensor:
    """
    Compute cross-entropy loss with optional z-loss regularization.

    This function expects logits that have already been materialized. It uses
    Greyhound's standalone cross-entropy kernel to compute the summed loss and
    logits gradient in one streaming pass, then exposes normal autograd behavior
    for ``logits``. Unlike the raw kernel helper, this functional wrapper does
    not mutate the input ``logits`` tensor.

    Args:
        logits: Contiguous logits tensor of shape ``[B, V]``.
        target: Target labels of shape ``[B]`` containing class indices in
            ``[0, V)``.
        ignore_index: Target value to ignore when computing the loss. Tokens
            with this target value will not contribute to the loss or gradient.
            Default: ``-100``.
        reduction: Specifies the reduction to apply to the output. One of:
            - ``"mean"``: divide the summed loss by the number of valid tokens
            - ``"sum"``: return the summed loss
            Default: ``"mean"``.
        z_loss_multiplier: Coefficient for the auxiliary z-loss term, which
            penalizes large log-partition function values. Set to ``0.0`` to
            disable. Default: ``0.0``.

    Returns:
        The combined cross-entropy loss plus z-loss scaled by
        ``z_loss_multiplier``.
    """
    return cast(
        Tensor,
        CrossEntropyFunction.apply(
            logits,
            target,
            ignore_index,
            reduction,
            z_loss_multiplier,
        ),
    )


def autograd_loss_and_logits_grad(
    loss_fn: Callable[..., Tensor],
) -> Callable[..., tuple[Tensor, Tensor]]:
    """
    Adapt a scalar PyTorch loss function to return ``(loss, grad_logits)``.

    ``chunked_linear_loss`` expects a callable that returns both a scalar loss and
    the gradient of that scalar with respect to the logits chunk. This helper wraps
    ordinary differentiable loss functions, such as ``torch.nn.functional.cross_entropy``
    or ``torch.nn.functional.mse_loss``, by running PyTorch autograd on the logits
    argument inside each chunk.

    Args:
        loss_fn: Callable receiving ``(logits_chunk, *args, **kwargs)`` and returning
            a scalar tensor differentiable with respect to ``logits_chunk``.

    Returns:
        Callable receiving the same arguments as ``loss_fn`` and returning
        ``(loss, grad_logits)``.
    """

    def loss_and_grad(logits: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            logits_for_grad = logits.detach().requires_grad_(True)
            loss = loss_fn(logits_for_grad, *args, **kwargs)
            _check_scalar_loss(loss)
            (grad_logits,) = torch.autograd.grad(loss, logits_for_grad)
        return loss.detach(), grad_logits.detach()

    return loss_and_grad


def chunked_linear_loss(
    inputs: Tensor,
    weight: Tensor,
    loss_and_grad_fn: Callable[..., tuple[Tensor, Tensor]],
    *loss_args: Any,
    chunk_size: int | None = None,
    grad_weight_accum_dtype: Literal["fp32", "weight"] = "fp32",
    **loss_kwargs: Any,
) -> Tensor:
    """
    Compute a chunked linear projection followed by a user-provided loss and logits gradient.

    This function evaluates ``inputs @ weight.T`` in row chunks. For each chunk,
    it calls ``loss_and_grad_fn(logits_chunk, *chunk_args, **chunk_kwargs)``.
    Tensor arguments whose leading dimension matches ``inputs.shape[0]`` are
    sliced to the same row range as the logits chunk; other arguments are passed
    through unchanged. The loss-and-gradient function must return
    ``(loss, grad_logits)``, where ``loss`` is a scalar tensor and
    ``grad_logits`` has the same shape as ``logits_chunk``. Use
    ``autograd_loss_and_logits_grad`` to adapt ordinary PyTorch scalar loss
    functions to this contract.

    Gradients for ``inputs`` and ``weight`` are accumulated chunk-by-chunk from
    the per-chunk logits gradient, avoiding materialization of the full logits
    tensor.

    Args:
        inputs: Input tensor of shape ``[B, D]``.
        weight: Linear weight matrix of shape ``[V, D]``.
        loss_and_grad_fn: Callable receiving
            ``(logits_chunk, *chunk_args, **chunk_kwargs)`` and returning
            ``(loss, grad_logits)`` for that chunk.
        loss_args: Additional positional arguments for ``loss_and_grad_fn``. Tensor
            arguments with leading dimension ``B`` are sliced per chunk.
        chunk_size: Optional number of rows per logits chunk. Defaults to an
            internal memory-aware heuristic.
        grad_weight_accum_dtype: Accumulator dtype for the chunked weight-gradient
            reduction. ``"fp32"`` accumulates partial ``inputs.T @ grad_logits``
            products in fp32 and casts once before returning the gradient.
            ``"weight"`` accumulates directly in the parameter dtype.
            Default: ``"fp32"``.
        loss_kwargs: Additional keyword arguments for ``loss_and_grad_fn``. Tensor
            values with leading dimension ``B`` are sliced per chunk.

    Returns:
        Sum of the scalar losses returned by ``loss_and_grad_fn`` for each chunk.
    """
    if grad_weight_accum_dtype not in ("fp32", "weight"):
        raise ValueError(
            "grad_weight_accum_dtype must be either 'fp32' or 'weight', "
            f"got {grad_weight_accum_dtype!r}"
        )

    batch_size = inputs.shape[0]

    def chunk_loss_and_grad_fn(logits_chunk: Tensor, start: int, end: int) -> tuple[Tensor, Tensor]:
        chunk_args = tuple(_slice_chunk_arg(arg, start, end, batch_size) for arg in loss_args)
        chunk_kwargs = {
            key: _slice_chunk_arg(value, start, end, batch_size)
            for key, value in loss_kwargs.items()
        }
        return loss_and_grad_fn(logits_chunk, *chunk_args, **chunk_kwargs)

    return cast(
        Tensor,
        ChunkedLinearLossFunction.apply(
            inputs,
            weight.T,
            chunk_loss_and_grad_fn,
            chunk_size,
            grad_weight_accum_dtype == "fp32",
        ),
    )


def chunked_linear_cross_entropy(
    inputs: Tensor,
    weight: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: Literal["sum", "mean"] = "mean",
    z_loss_multiplier: float = 0.0,
    grad_weight_accum_dtype: Literal["fp32", "weight"] = "fp32",
) -> Tensor:
    """
    Compute chunked linear cross-entropy with optional z-loss regularization.

    This is a specialization of ``chunked_linear_loss`` for cross-entropy. It uses
    the same chunked linear-gradient accumulation and a fused
    cross-entropy-and-logits-gradient kernel per logits tile, avoiding materialization
    of the full logits tensor in memory.
    During training, the weight gradient is accumulated in fp32 by default and cast once
    at the end. That is the conservative mixed-precision choice: the matmuls run in
    bf16/fp16, while the chunked reduction over tokens keeps fp32 accumulator state.
    Set ``grad_weight_accum_dtype="weight"`` to accumulate directly in the weight dtype
    for lower memory and higher speed when that numerical tradeoff is acceptable.

    Args:
        inputs: Input tensor of shape [BT, D] where BT is the number of tokens and D is
            the hidden dimension.
        weight: Weight matrix of shape [V, D] where V is the vocabulary size.
        target: Target labels of shape [BT] containing class indices in [0, V).
        ignore_index: Target value to ignore when computing the loss. Tokens with this
            target value will not contribute to the loss or gradient. Default: -100.
        reduction: Specifies the reduction to apply to the output. One of:
            - "mean": the sum of the output will be divided by the number of valid tokens
            - "sum": the output will be summed
            Default: "mean".
        z_loss_multiplier: Coefficient for the auxiliary z-loss term, which penalizes
            large log-partition function values (log-sum-exp of logits). Set to 0.0 to disable.
            Default: 0.0.
        grad_weight_accum_dtype: Accumulator dtype for the chunked weight-gradient reduction.
            ``"fp32"`` accumulates partial ``x.T @ grad_logits`` products in fp32 and casts
            once before returning the gradient. ``"weight"`` accumulates directly in the
            parameter dtype, which is faster and uses less memory but rounds between chunks.
            Default: ``"fp32"``.

    Returns:
        The combined cross-entropy loss plus z-loss (scaled by z_loss_multiplier).
    """
    if grad_weight_accum_dtype not in ("fp32", "weight"):
        raise ValueError(
            "grad_weight_accum_dtype must be either 'fp32' or 'weight', "
            f"got {grad_weight_accum_dtype!r}"
        )
    return cast(
        Tensor,
        ChunkedLinearCrossEntropyFunction.apply(
            inputs,
            weight.T,
            target,
            ignore_index,
            reduction,
            z_loss_multiplier,
            grad_weight_accum_dtype == "fp32",
        ),
    )
