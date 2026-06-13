from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from greyhound.kernels.cross_entropy import cross_entropy_with_grad_kernel
from greyhound.ops.chunked_linear_loss import (
    _chunked_linear_loss_chunk_size,
    _chunked_linear_loss_with_grad,
)

__all__ = [
    "ChunkedLinearCrossEntropyFunction",
    "chunked_linear_cross_entropy_fwd",
]


@torch.library.custom_op("greyhound::chunked_linear_cross_entropy_fwd", mutates_args=())
def chunked_linear_cross_entropy_fwd(
    inputs: Tensor,  # [B, D]
    weight: Tensor,  # [D, V]
    target: Tensor,  # [B]
    z_loss_multiplier: float,
    fp32_grad_weight_accum: bool,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Chunked linear cross-entropy forward.

    Chunks over the batch dimension, using cuBLAS for the matmul and a CuTe DSL
    kernel for CE + gradient per chunk. Gradients for input and weight are
    computed inline to avoid materializing the full [B, V] logit tensor.
    By default, grad_weight is accumulated in fp32 and cast once at the end.
    This matches the usual mixed-precision training convention: matmuls run in
    bf16/fp16, but sensitive reductions accumulate in fp32.

    Returns (ce_sum, z_sum, n_valid, grad_input, grad_weight).
    """
    ce_sum = torch.zeros([], dtype=torch.float32, device=inputs.device)
    z_sum = torch.zeros([], dtype=torch.float32, device=inputs.device)
    n_valid_total = torch.zeros([], dtype=torch.float32, device=inputs.device)

    def loss_and_grad(logits_chunk: Tensor, start: int, end: int) -> tuple[Tensor, Tensor]:
        nonlocal ce_sum, z_sum, n_valid_total
        target_chunk = target[start:end]  # [C]

        # CE + gradient kernel overwrites logits_chunk with dlogits in-place.
        ce_chunk, z_chunk, n_chunk = cross_entropy_with_grad_kernel(
            logits_chunk,
            target_chunk,
            z_loss_multiplier,
            ignore_index,
        )

        ce_sum = ce_sum + ce_chunk
        z_sum = z_sum + z_chunk
        n_valid_total = n_valid_total + n_chunk

        return ce_chunk + z_loss_multiplier * z_chunk, logits_chunk

    _, grad_input, grad_weight = _chunked_linear_loss_with_grad(
        inputs,
        weight,
        loss_and_grad,
        fp32_grad_weight_accum,
    )
    return ce_sum, z_sum, n_valid_total, grad_input, grad_weight


@chunked_linear_cross_entropy_fwd.register_fake
def _(
    inputs: Tensor,
    weight: Tensor,
    target: Tensor,
    z_loss_multiplier: float,
    fp32_grad_weight_accum: bool,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del target, z_loss_multiplier, fp32_grad_weight_accum, ignore_index
    B, D = inputs.shape
    V = weight.shape[1]
    return (
        torch.empty([], dtype=torch.float32, device=inputs.device),  # ce_sum
        torch.empty([], dtype=torch.float32, device=inputs.device),  # z_sum
        torch.empty([], dtype=torch.float32, device=inputs.device),  # n_valid
        torch.empty([B, D], dtype=inputs.dtype, device=inputs.device),  # grad_input
        torch.empty([D, V], dtype=weight.dtype, device=inputs.device),  # grad_weight
    )


def _forward_only_chunked_ce(
    inputs: Tensor,
    weight: Tensor,
    target: Tensor,
    ignore_index: int,
    reduction: str,
    z_loss_multiplier: float,
) -> Tensor:
    """Fast forward-only path: chunked matmul + torch.nn.functional.cross_entropy.

    Skips all gradient computation for inference / forward-only benchmarks.
    """
    B, D = inputs.shape
    V = weight.shape[1]

    chunk_size = _chunked_linear_loss_chunk_size(B, D, V, inputs.element_size())

    ce_sum = torch.zeros([], dtype=torch.float32, device=inputs.device)
    z_sum = torch.zeros([], dtype=torch.float32, device=inputs.device)
    n_valid_total = torch.zeros([], dtype=torch.float32, device=inputs.device)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        logits_chunk = inputs[start:end] @ weight  # [C, V]
        target_chunk = target[start:end]

        ce_sum = ce_sum + F.cross_entropy(
            logits_chunk.float(), target_chunk, ignore_index=ignore_index, reduction="sum"
        )

        valid_mask = target_chunk != ignore_index
        n_valid_total = n_valid_total + valid_mask.sum().float()

        if z_loss_multiplier > 0.0:
            lse = torch.logsumexp(logits_chunk.float(), dim=-1)
            z_sum = z_sum + (lse.pow(2) * valid_mask).sum()

    loss = ce_sum + z_loss_multiplier * z_sum
    if reduction == "mean":
        loss = loss / n_valid_total.clamp(min=1.0)

    return loss


class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,  # [B, D]
        weight: torch.Tensor,  # [D, V]
        target: torch.Tensor,  # [B]
        ignore_index: int = -100,
        reduction: Literal["sum", "mean"] = "mean",
        z_loss_multiplier: float = 0.0,
        fp32_grad_weight_accum: bool = True,
    ) -> torch.Tensor:
        # Fast path: skip gradient computation when no input requires grad.
        # Note: torch.is_grad_enabled() is always False inside autograd Function.forward(),
        # so we must not check it here. Use ctx.needs_input_grad instead.
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:  # ty:ignore[unresolved-attribute]
            return _forward_only_chunked_ce(
                inputs, weight, target, ignore_index, reduction, z_loss_multiplier
            )

        ce_sum, z_sum, n_valid, grad_input, grad_weight = chunked_linear_cross_entropy_fwd(
            inputs, weight, target, z_loss_multiplier, fp32_grad_weight_accum, ignore_index
        )

        loss = ce_sum + z_loss_multiplier * z_sum
        if reduction == "mean":
            safe_n_valid = n_valid.clamp(min=1.0)
            loss = loss / safe_n_valid
            grad_input.mul_(safe_n_valid.reciprocal())
            grad_weight.mul_(safe_n_valid.reciprocal())
        elif reduction != "sum":
            raise ValueError(f"Unsupported reduction: {reduction}")

        ctx.save_for_backward(grad_input, grad_weight, n_valid)  # ty:ignore[unresolved-attribute]
        return loss

    @staticmethod
    def backward(
        ctx: object, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:  # ty:ignore[invalid-method-override]
        grad_input, grad_weight, _ = ctx.saved_tensors  # ty:ignore[unresolved-attribute]

        return (
            grad_input * grad_output,
            grad_weight * grad_output,
            None,  # target
            None,  # ignore_index
            None,  # reduction
            None,  # z_loss_multiplier
            None,  # fp32_grad_weight_accum
        )
