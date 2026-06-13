from typing import Literal, cast

import pytest
import torch
from torch import Tensor
from torch.library import opcheck

from greyhound.nn.functional import (
    autograd_loss_and_logits_grad,
    chunked_linear_cross_entropy,
    chunked_linear_loss,
)
from greyhound.ops.chunked_linear_cross_entropy import chunked_linear_cross_entropy_fwd
from greyhound.testing import requires_gpu
from greyhound.utils import get_default_device


def reference_linear_cross_entropy(
    inputs: Tensor,
    weight: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 0.0,
) -> Tensor:
    """Reference implementation using standard PyTorch ops."""
    # weight is [D, V] in our convention, so logits = inputs @ weight
    logits = inputs @ weight  # [B, V]
    ce_loss = torch.nn.functional.cross_entropy(
        logits.float(), target, ignore_index=ignore_index, reduction=reduction
    )

    if z_loss_multiplier > 0.0:
        # z_loss = z_loss_multiplier * mean(lse^2)
        lse = torch.logsumexp(logits, dim=-1)  # [B]
        if ignore_index >= 0:
            mask = target != ignore_index
            lse = lse[mask]
        z_loss = z_loss_multiplier * (lse**2).mean()
        return ce_loss + z_loss

    return ce_loss


def _assert_chunked_loss_matches_reference(
    result: Tensor,
    expected: Tensor,
    inputs: Tensor,
    weight: Tensor,
    inputs_ref: Tensor,
    weight_ref: Tensor,
    tolerance: float,
) -> None:
    assert torch.allclose(result.float(), expected.float(), rtol=5e-3, atol=5e-3)

    result.backward()
    expected.backward()

    assert inputs.grad is not None and weight.grad is not None
    assert inputs_ref.grad is not None and weight_ref.grad is not None

    input_rel_l2 = (
        inputs.grad.float() - inputs_ref.grad.float()
    ).norm() / inputs_ref.grad.float().norm()
    weight_rel_l2 = (
        weight.grad.float() - weight_ref.grad.float()
    ).norm() / weight_ref.grad.float().norm()
    assert input_rel_l2 < tolerance, f"Input gradient relative L2 error too large: {input_rel_l2}"
    assert weight_rel_l2 < tolerance, (
        f"Weight gradient relative L2 error too large: {weight_rel_l2}"
    )


@requires_gpu
@pytest.mark.parametrize(
    ("loss_name", "grad_weight_accum_dtype"),
    [
        ("cross_entropy", "fp32"),
        ("nll_loss_after_log_softmax", "fp32"),
        ("mse_loss", "fp32"),
        ("mse_loss", "weight"),
        ("smooth_l1_loss", "fp32"),
        ("binary_cross_entropy_with_logits", "fp32"),
        ("custom_weighted_mean_square", "fp32"),
        ("custom_weighted_mean_square_direct_grad", "fp32"),
    ],
)
def test_chunked_linear_loss_with_various_loss_functions(
    loss_name: str,
    grad_weight_accum_dtype: Literal["fp32", "weight"],
) -> None:
    """Test chunked_linear_loss with existing and custom loss implementations."""
    device = get_default_device()
    batch_size, hidden_dim, out_dim = 37, 64, 257
    inputs = torch.randn(
        batch_size, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        out_dim, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    inputs_ref = inputs.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    logits_ref = (inputs_ref @ weight_ref.T).float()

    chunked_kwargs: dict[str, object] = {
        "chunk_size": 11,
        "grad_weight_accum_dtype": grad_weight_accum_dtype,
    }

    if loss_name == "cross_entropy":
        class_target = torch.randint(0, out_dim, (batch_size,), device=device)
        result = chunked_linear_loss(
            inputs,
            weight,
            autograd_loss_and_logits_grad(torch.nn.functional.cross_entropy),
            class_target,
            reduction="sum",
            **chunked_kwargs,
        )
        expected = torch.nn.functional.cross_entropy(logits_ref, class_target, reduction="sum")
    elif loss_name == "nll_loss_after_log_softmax":
        class_target = torch.randint(0, out_dim, (batch_size,), device=device)

        def nll_loss_after_log_softmax(logits: Tensor, target: Tensor) -> Tensor:
            return torch.nn.functional.nll_loss(
                logits.float().log_softmax(dim=-1), target, reduction="sum"
            )

        result = chunked_linear_loss(
            inputs,
            weight,
            autograd_loss_and_logits_grad(nll_loss_after_log_softmax),
            class_target,
            **chunked_kwargs,
        )
        expected = nll_loss_after_log_softmax(logits_ref, class_target)
    elif loss_name == "mse_loss":
        regression_target = torch.randn(batch_size, out_dim, device=device, dtype=torch.float32)

        def mse_loss(logits: Tensor, target: Tensor, reduction: str) -> Tensor:
            return torch.nn.functional.mse_loss(logits.float(), target, reduction=reduction)

        result = chunked_linear_loss(
            inputs,
            weight,
            autograd_loss_and_logits_grad(mse_loss),
            regression_target,
            reduction="sum",
            **chunked_kwargs,
        )
        expected = mse_loss(logits_ref, regression_target, reduction="sum")
    elif loss_name == "smooth_l1_loss":
        regression_target = torch.randn(batch_size, out_dim, device=device, dtype=torch.float32)
        result = chunked_linear_loss(
            inputs,
            weight,
            autograd_loss_and_logits_grad(torch.nn.functional.smooth_l1_loss),
            target=regression_target,
            reduction="sum",
            beta=0.25,
            **chunked_kwargs,
        )
        expected = torch.nn.functional.smooth_l1_loss(
            logits_ref, target=regression_target, reduction="sum", beta=0.25
        )
    elif loss_name == "binary_cross_entropy_with_logits":
        binary_target = torch.rand(batch_size, out_dim, device=device, dtype=torch.float32)
        positive_weight = torch.rand(out_dim, device=device, dtype=torch.float32) + 0.5
        result = chunked_linear_loss(
            inputs,
            weight,
            autograd_loss_and_logits_grad(torch.nn.functional.binary_cross_entropy_with_logits),
            binary_target,
            reduction="sum",
            pos_weight=positive_weight,
            **chunked_kwargs,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_ref, binary_target, reduction="sum", pos_weight=positive_weight
        )
    elif loss_name == "custom_weighted_mean_square":
        regression_target = torch.randn(batch_size, out_dim, device=device, dtype=torch.float32)
        row_weight = torch.rand(batch_size, device=device, dtype=torch.float32) + 0.5
        feature_weight = torch.rand(out_dim, device=device, dtype=torch.float32) + 0.5

        def weighted_mean_square(
            logits: Tensor,
            target: Tensor,
            row_weight: Tensor,
            feature_weight: Tensor,
            scale: float,
        ) -> Tensor:
            error = logits.float() - target
            return scale * (error.square() * row_weight[:, None] * feature_weight[None, :]).sum()

        result = chunked_linear_loss(
            inputs,
            weight,
            autograd_loss_and_logits_grad(weighted_mean_square),
            regression_target,
            row_weight,
            feature_weight,
            scale=0.125,
            **chunked_kwargs,
        )
        expected = weighted_mean_square(
            logits_ref, regression_target, row_weight, feature_weight, scale=0.125
        )
    elif loss_name == "custom_weighted_mean_square_direct_grad":
        regression_target = torch.randn(batch_size, out_dim, device=device, dtype=torch.float32)
        row_weight = torch.rand(batch_size, device=device, dtype=torch.float32) + 0.5
        feature_weight = torch.rand(out_dim, device=device, dtype=torch.float32) + 0.5

        def weighted_mean_square_loss_and_grad(
            logits: Tensor,
            target: Tensor,
            row_weight: Tensor,
            feature_weight: Tensor,
            scale: float,
        ) -> tuple[Tensor, Tensor]:
            error = logits.float() - target
            weights = row_weight[:, None] * feature_weight[None, :]
            loss = scale * (error.square() * weights).sum()
            grad_logits = (2 * scale * error * weights).to(logits.dtype)
            return loss, grad_logits

        result = chunked_linear_loss(
            inputs,
            weight,
            weighted_mean_square_loss_and_grad,
            regression_target,
            row_weight,
            feature_weight,
            scale=0.125,
            **chunked_kwargs,
        )
        error = logits_ref - regression_target
        expected = 0.125 * (error.square() * row_weight[:, None] * feature_weight[None, :]).sum()
    else:
        raise AssertionError(f"unknown loss case: {loss_name}")

    tolerance = 5e-2 if grad_weight_accum_dtype == "fp32" else 8e-2
    _assert_chunked_loss_matches_reference(
        result,
        expected,
        inputs,
        weight,
        inputs_ref,
        weight_ref,
        tolerance,
    )


@requires_gpu
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("ignore_index", [None, -100])
@pytest.mark.parametrize("z_loss_multiplier", [0.0, 1e-4])
def test_chunked_linear_cross_entropy_correctness(
    reduction: str, ignore_index: int | None, z_loss_multiplier: float
) -> None:
    """Test that chunked_linear_cross_entropy produces correct output."""
    device = get_default_device()
    batch_size, hidden_dim, vocab_size = 32, 128, 1000

    inputs = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_dim, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, vocab_size, (batch_size,), device=device)

    # Set some targets to ignore_index if specified
    if ignore_index is not None:
        target[::4] = ignore_index

    actual_ignore_index = ignore_index if ignore_index is not None else -100

    result = chunked_linear_cross_entropy(
        inputs,
        weight,
        target,
        ignore_index=actual_ignore_index,
        reduction="mean" if reduction == "mean" else "sum",
        z_loss_multiplier=z_loss_multiplier,
    )
    expected = reference_linear_cross_entropy(
        inputs,
        weight.T,
        target,
        ignore_index=actual_ignore_index,
        reduction=reduction,
        z_loss_multiplier=z_loss_multiplier,
    )

    assert torch.allclose(result.float(), expected.float(), rtol=5e-3, atol=5e-3)


@requires_gpu
@pytest.mark.parametrize("z_loss_multiplier", [0.0, 1e-4])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_chunked_linear_cross_entropy_gradient_correctness(
    reduction: str, z_loss_multiplier: float
) -> None:
    """Test that gradients flow correctly through chunked_linear_cross_entropy."""
    device = get_default_device()
    batch_size, hidden_dim, vocab_size = 32, 128, 1000

    inputs = torch.randn(
        batch_size, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        vocab_size, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.randint(0, vocab_size, (batch_size,), device=device)

    # Reference inputs
    inputs_ref = inputs.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)

    # Forward
    result = chunked_linear_cross_entropy(
        inputs,
        weight,
        target,
        reduction=cast(Literal["sum", "mean"], reduction),
        z_loss_multiplier=z_loss_multiplier,
    )
    expected = reference_linear_cross_entropy(
        inputs_ref, weight_ref.T, target, reduction=reduction, z_loss_multiplier=z_loss_multiplier
    )

    # Backward
    result.backward()
    expected.backward()

    assert inputs.grad is not None and weight.grad is not None
    assert inputs_ref.grad is not None and weight_ref.grad is not None
    # Use relative L2 norm: element-wise allclose fails on sum reduction because gradient
    # magnitudes are unbounded, causing outlier elements to exceed any fixed atol.
    input_rel_l2 = (
        inputs.grad.float() - inputs_ref.grad.float()
    ).norm() / inputs_ref.grad.float().norm()
    weight_rel_l2 = (
        weight.grad.float() - weight_ref.grad.float()
    ).norm() / weight_ref.grad.float().norm()
    assert input_rel_l2 < 5e-2, f"Input gradient relative L2 error too large: {input_rel_l2:.4e}"
    assert weight_rel_l2 < 5e-2, f"Weight gradient relative L2 error too large: {weight_rel_l2:.4e}"


@requires_gpu
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.bfloat16, id="bfloat16"),
        pytest.param(torch.float32, id="float32"),
    ],
)
def test_chunked_linear_cross_entropy_fwd_opcheck(dtype: torch.dtype) -> None:
    """Test chunked_linear_cross_entropy_fwd custom op with opcheck."""
    device = get_default_device()

    configs = [
        (16, 64, 256),
        (32, 128, 1000),
        (64, 256, 2000),
    ]

    for batch_size, hidden_dim, vocab_size in configs:
        inputs = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
        weight = torch.randn(hidden_dim, vocab_size, device=device, dtype=dtype)
        target = torch.randint(0, vocab_size, (batch_size,), device=device)

        opcheck(
            chunked_linear_cross_entropy_fwd,
            (inputs, weight, target, 0.0, True, -100),
            raise_exception=True,
        )
        opcheck(
            chunked_linear_cross_entropy_fwd,
            (inputs, weight, target, 1e-4, True, -100),
            raise_exception=True,
        )
        opcheck(
            chunked_linear_cross_entropy_fwd,
            (inputs, weight, target, 0.0, False, -100),
            raise_exception=True,
        )


@requires_gpu
@pytest.mark.parametrize(
    "z_loss_multiplier",
    [pytest.param(0.0, id="no_z_loss"), pytest.param(1e-4, id="with_z_loss")],
)
def test_chunked_linear_cross_entropy_compile(z_loss_multiplier: float) -> None:
    """Test that chunked_linear_cross_entropy works with torch.compile."""
    device = get_default_device()
    batch_size, hidden_dim, vocab_size = 32, 128, 1000

    @torch.compile(fullgraph=True)
    def compiled_fn(inputs: Tensor, weight: Tensor, target: Tensor) -> Tensor:
        return chunked_linear_cross_entropy(
            inputs, weight, target, z_loss_multiplier=z_loss_multiplier
        )

    inputs = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_dim, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, vocab_size, (batch_size,), device=device)

    result = compiled_fn(inputs, weight, target)
    expected = reference_linear_cross_entropy(
        inputs, weight.T, target, z_loss_multiplier=z_loss_multiplier
    )

    assert torch.allclose(result.float(), expected.float(), rtol=5e-3, atol=5e-3)
