from typing import Literal

import torch
import torch.nn.functional as F

from greyhound.bonus.newton_schultz import DEFAULT_NS_CONSTS


def torch_linear_cross_entropy(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    logits = inputs @ weight.T
    ce_loss = torch.nn.functional.cross_entropy(
        logits.float(), target, ignore_index=ignore_index, reduction=reduction
    )
    if z_loss_multiplier > 0.0:
        lse = torch.logsumexp(logits.float(), dim=-1)
        if reduction == "mean":
            mask = target != ignore_index
            z_loss = z_loss_multiplier * (lse[mask] ** 2).mean()
        else:
            mask = target != ignore_index
            z_loss = z_loss_multiplier * (lse[mask] ** 2).sum()
        return ce_loss + z_loss
    return ce_loss


def torch_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    """Standalone cross-entropy on pre-computed logits (no linear projection)."""
    ce_loss = F.cross_entropy(
        logits.float(), target, ignore_index=ignore_index, reduction=reduction
    )
    if z_loss_multiplier > 0:
        lse = torch.logsumexp(logits.float(), dim=-1)
        mask = target != ignore_index
        z_loss = (lse[mask] ** 2).mean() if reduction == "mean" else (lse[mask] ** 2).sum()
        return ce_loss + z_loss_multiplier * z_loss
    return ce_loss


def torch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: Literal["silu", "swish"] | None = None,
) -> torch.Tensor:
    _, D, T = x.shape
    _D, W = weight.shape
    assert D == _D, "Input and weight dimensions must match"
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :T]
    if activation in {"silu", "swish"}:
        return F.silu(out)
    return out


def torch_selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)  # shape: (batch_size, seq_len, vocab_size)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def torch_newton_schulz(
    g: torch.Tensor,
    epsilon: float = 1e-7,
    ns_consts: tuple[tuple[float, float, float], ...] | None = None,
) -> torch.Tensor:
    if ns_consts is None:
        ns_consts = DEFAULT_NS_CONSTS
    x = g.to(dtype=torch.bfloat16)
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in ns_consts:
        gram = x @ x.mT
        if x.ndim == 3:
            update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.baddbmm(x, update, x, beta=a, alpha=1.0)
        else:
            update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.addmm(x, update, x, beta=a, alpha=1.0)

    if transposed:
        x = x.mT
    return x
