from typing import Literal

import pytest
import torch
from torch import Tensor
from torch.library import opcheck

from greyhound.nn.functional import cross_entropy
from greyhound.ops.cross_entropy import cross_entropy_fwd
from greyhound.testing import requires_gpu
from greyhound.utils import get_default_device


def reference_cross_entropy(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: Literal["sum", "mean"] = "mean",
    z_loss_multiplier: float = 0.0,
) -> Tensor:
    logits = logits.float()
    ce_loss = torch.nn.functional.cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if z_loss_multiplier == 0.0:
        return ce_loss

    valid = target != ignore_index
    z_sum = torch.logsumexp(logits, dim=-1).square()[valid].sum()
    z_loss = z_loss_multiplier * z_sum
    if reduction == "mean":
        z_loss = z_loss / valid.sum().clamp(min=1)
    return ce_loss + z_loss


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("z_loss_multiplier", [0.0, 1e-4])
def test_cross_entropy_correctness_and_grad(
    dtype: torch.dtype,
    reduction: Literal["sum", "mean"],
    z_loss_multiplier: float,
) -> None:
    device = get_default_device()
    batch_size, vocab_size = 37, 257
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size,), device=device)
    target[::5] = -100
    logits_ref = logits.detach().clone().requires_grad_(True)

    result = cross_entropy(
        logits,
        target,
        reduction=reduction,
        z_loss_multiplier=z_loss_multiplier,
    )
    expected = reference_cross_entropy(
        logits_ref,
        target,
        reduction=reduction,
        z_loss_multiplier=z_loss_multiplier,
    )

    assert torch.allclose(result.float(), expected.float(), rtol=5e-3, atol=5e-3)

    result.backward()
    expected.backward()

    assert logits.grad is not None
    assert logits_ref.grad is not None
    grad_rel_l2 = (
        logits.grad.float() - logits_ref.grad.float()
    ).norm() / logits_ref.grad.float().norm()
    tolerance = 5e-2 if dtype is torch.bfloat16 else 5e-3
    assert grad_rel_l2 < tolerance, f"logits gradient relative L2 error too large: {grad_rel_l2}"


@requires_gpu
def test_cross_entropy_does_not_mutate_logits() -> None:
    device = get_default_device()
    logits = torch.randn(8, 97, device=device, dtype=torch.bfloat16, requires_grad=True)
    original = logits.detach().clone()
    target = torch.randint(0, logits.shape[-1], (logits.shape[0],), device=device)

    loss = cross_entropy(logits, target)
    loss.backward()

    assert torch.equal(logits.detach(), original)


@requires_gpu
def test_cross_entropy_fwd_returns_logits_gradient() -> None:
    device = get_default_device()
    logits = torch.randn(16, 257, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, logits.shape[-1], (logits.shape[0],), device=device)
    target[::4] = -100
    logits_ref = logits.detach().clone().float().requires_grad_(True)

    ce_sum, z_sum, n_valid, grad_logits = cross_entropy_fwd(
        logits,
        target,
        1e-4,
        -100,
    )
    expected = reference_cross_entropy(
        logits_ref,
        target,
        reduction="sum",
        z_loss_multiplier=1e-4,
    )
    expected.backward()

    assert torch.allclose((ce_sum + 1e-4 * z_sum).float(), expected.detach(), rtol=5e-3, atol=5e-3)
    assert n_valid.item() == float((target != -100).sum().item())
    assert logits_ref.grad is not None
    assert torch.allclose(grad_logits.float(), logits_ref.grad, rtol=5e-3, atol=5e-3)


@requires_gpu
def test_cross_entropy_fwd_opcheck() -> None:
    device = get_default_device()
    logits = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, logits.shape[-1], (logits.shape[0],), device=device)

    opcheck(cross_entropy_fwd, (logits, target, 0.0, -100), raise_exception=True)
    opcheck(cross_entropy_fwd, (logits, target, 1e-4, -100), raise_exception=True)


@requires_gpu
@pytest.mark.parametrize("z_loss_multiplier", [0.0, 1e-4])
def test_cross_entropy_compile(z_loss_multiplier: float) -> None:
    device = get_default_device()
    logits = torch.randn(16, 257, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, logits.shape[-1], (logits.shape[0],), device=device)

    @torch.compile(fullgraph=True)
    def compiled_fn(logits: Tensor, target: Tensor) -> Tensor:
        return cross_entropy(logits, target, z_loss_multiplier=z_loss_multiplier)

    result = compiled_fn(logits, target)
    expected = reference_cross_entropy(logits, target, z_loss_multiplier=z_loss_multiplier)

    assert torch.allclose(result.float(), expected.float(), rtol=5e-3, atol=5e-3)
