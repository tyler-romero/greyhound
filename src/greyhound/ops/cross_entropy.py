from typing import Any, Literal

import torch
from torch import Tensor

from greyhound.kernels.cross_entropy import cross_entropy_with_grad_kernel

__all__ = ["CrossEntropyFunction", "cross_entropy_fwd"]


@torch.library.custom_op("greyhound::cross_entropy_fwd", mutates_args=())
def cross_entropy_fwd(
    logits: Tensor,
    target: Tensor,
    z_loss_multiplier: float,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cross-entropy forward that also returns the logits gradient."""
    grad_logits = logits.detach().clone()
    ce_sum, z_sum, n_valid = cross_entropy_with_grad_kernel(
        grad_logits,
        target,
        z_loss_multiplier,
        ignore_index,
    )
    return ce_sum, z_sum, n_valid, grad_logits


@cross_entropy_fwd.register_fake
def _(
    logits: Tensor,
    target: Tensor,
    z_loss_multiplier: float,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del target, z_loss_multiplier, ignore_index
    return (
        torch.empty([], dtype=torch.float32, device=logits.device),
        torch.empty([], dtype=torch.float32, device=logits.device),
        torch.empty([], dtype=torch.float32, device=logits.device),
        torch.empty_like(logits),
    )


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: Tensor,
        target: Tensor,
        ignore_index: int = -100,
        reduction: Literal["sum", "mean"] = "mean",
        z_loss_multiplier: float = 0.0,
    ) -> Tensor:
        ce_sum, z_sum, n_valid, grad_logits = cross_entropy_fwd(
            logits,
            target,
            z_loss_multiplier,
            ignore_index,
        )

        loss = ce_sum + z_loss_multiplier * z_sum
        if reduction == "mean":
            safe_n_valid = n_valid.clamp(min=1.0)
            loss = loss / safe_n_valid
            grad_logits.mul_(safe_n_valid.reciprocal())
        elif reduction != "sum":
            raise ValueError(f"Unsupported reduction: {reduction}")

        ctx.save_for_backward(grad_logits)
        return loss

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None, None]:  # ty:ignore[invalid-method-override]
        (grad_logits,) = ctx.saved_tensors
        return (
            grad_logits * grad_output,
            None,  # target
            None,  # ignore_index
            None,  # reduction
            None,  # z_loss_multiplier
        )
