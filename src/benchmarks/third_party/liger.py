import torch

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.functional import (
        liger_fused_linear_cross_entropy,
    )

    has_liger = True
except ImportError:
    has_liger = False


__all__ = [
    "has_liger",
    "liger_cross_entropy",
    "liger_linear_cross_entropy",
]


@torch.compiler.disable()
def liger_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    """Standalone cross-entropy on pre-computed logits using Liger Kernel."""
    assert has_liger, "Liger Kernel is not installed"
    loss_fn = LigerCrossEntropyLoss(
        ignore_index=ignore_index,
        lse_square_scale=z_loss_multiplier,
        reduction=reduction,
    )
    return loss_fn(logits, target)


@torch.compiler.disable()
def liger_linear_cross_entropy(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 1e-3,
) -> torch.Tensor:
    assert has_liger, "Liger Kernel is not installed"
    out = liger_fused_linear_cross_entropy(
        inputs,
        weight,
        target,
        ignore_index=ignore_index,
        lse_square_scale=z_loss_multiplier,
        reduction=reduction,
        return_z_loss=True,
        accum_dtype=torch.float32,
    )
    return out.loss
