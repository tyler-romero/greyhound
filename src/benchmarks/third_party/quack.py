from typing import Literal

import torch

try:
    from quack.cross_entropy import cross_entropy_fwd as quack_cross_entropy_fwd
    from quack.linear_cross_entropy import (
        chunked_linear_cross_entropy as quack_chunked_linear_cross_entropy,
    )

    has_quack = True
except ImportError:
    has_quack = False

__all__ = [
    "has_quack",
    "quack_cross_entropy",
    "quack_linear_cross_entropy",
]


def quack_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    """Standalone cross-entropy using Quack, overwriting logits with dlogits."""
    assert quack_cross_entropy_fwd is not None, "Quack is not installed"
    if z_loss_multiplier != 0.0:
        raise ValueError("Quack cross_entropy provider does not support z-loss")

    loss, _ = quack_cross_entropy_fwd(  # ty:ignore[invalid-assignment]
        logits,
        target,
        ignore_index=ignore_index,
        return_dx=True,
        inplace_backward=True,
    )
    loss_sum = loss.sum()
    if reduction == "mean":
        return loss_sum / (target != ignore_index).sum().clamp(min=1).float()
    return loss_sum


def quack_linear_cross_entropy(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum"] = "sum",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    """Chunked linear cross-entropy using Quack."""
    assert has_quack, "Quack is not installed"
    if z_loss_multiplier != 0.0:
        raise ValueError("Quack linear_cross_entropy provider does not support z-loss")

    return quack_chunked_linear_cross_entropy(
        inputs, weight, target, ignore_index=ignore_index, reduction=reduction
    )
