import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import opcheck

from greyhound.nn.functional import causal_conv1d
from greyhound.ops.causal_conv1d import causal_conv1d_bwd, causal_conv1d_fwd
from greyhound.testing import requires_gpu
from greyhound.utils import get_default_device


def reference_causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str | None = None,
) -> Tensor:
    """Reference implementation using standard PyTorch ops."""
    _, D, _ = x.shape
    _D, W = weight.shape
    # F.conv1d expects weight shape [out_channels, in_channels/groups, kW]
    # For depthwise: groups=D, so weight is [D, 1, W]
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., : x.shape[-1]]
    if activation == "silu":
        return F.silu(out)
    return out


@requires_gpu
@pytest.mark.parametrize(
    "shape",
    [(2, 64, 128, 4), (4, 128, 256, 2), (1, 256, 512, 3)],
)
@pytest.mark.parametrize("activation", [None, "silu"])
def test_causal_conv1d_correctness(
    shape: tuple[int, int, int, int], activation: str | None
) -> None:
    """Test that causal_conv1d produces correct output compared to reference."""
    device = get_default_device()
    B, D, T, W = shape
    x = torch.randn(B, D, T, device=device, dtype=torch.bfloat16)
    weight = torch.randn(D, W, device=device, dtype=torch.bfloat16)
    bias = torch.randn(D, device=device, dtype=torch.bfloat16)

    result = causal_conv1d(x, weight, bias, activation=activation)
    expected = reference_causal_conv1d(x, weight, bias, activation=activation)
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), (
        f"Failed for shape {shape}, activation={activation}"
    )


@requires_gpu
@pytest.mark.parametrize("activation", [None, "silu"])
def test_causal_conv1d_no_bias(activation: str | None) -> None:
    """Test causal_conv1d without bias."""
    device = get_default_device()
    x = torch.randn(2, 64, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, 4, device=device, dtype=torch.bfloat16)

    result = causal_conv1d(x, weight, bias=None, activation=activation)
    expected = reference_causal_conv1d(x, weight, bias=None, activation=activation)
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_causal_conv1d_gradient(dtype: torch.dtype, activation: str | None) -> None:
    """Test that gradients flow correctly through causal_conv1d."""
    device = get_default_device()
    x = torch.randn(2, 64, 128, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(64, 4, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(64, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)
    bias_ref = bias.clone().detach().requires_grad_(True)

    result = causal_conv1d(x, weight, bias, activation=activation)
    expected = reference_causal_conv1d(x_ref, weight_ref, bias_ref, activation=activation)
    grad_out = torch.randn_like(result)
    result.backward(grad_out)
    expected.backward(grad_out)

    tolerance = 1e-4 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(bias.grad, bias_ref.grad, rtol=tolerance, atol=tolerance)


@requires_gpu
def test_causal_conv1d_gradient_no_bias() -> None:
    """Test gradient flow without bias."""
    device = get_default_device()
    x = torch.randn(2, 64, 128, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(64, 4, device=device, dtype=torch.float32, requires_grad=True)

    x_ref = x.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)

    result = causal_conv1d(x, weight, bias=None, activation="silu")
    expected = reference_causal_conv1d(x_ref, weight_ref, bias=None, activation="silu")

    grad_out = torch.randn_like(result)
    result.backward(grad_out)
    expected.backward(grad_out)

    assert x.grad is not None and x_ref.grad is not None
    assert weight.grad is not None and weight_ref.grad is not None
    assert torch.allclose(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-4)


@requires_gpu
def test_causal_conv1d_fwd_opcheck() -> None:
    """Test causal_conv1d_fwd custom op with opcheck."""
    device = get_default_device()
    shapes = [
        (1, 32, 64, 2),
        (2, 64, 128, 4),
        (4, 128, 256, 3),
    ]
    for B, D, T, W in shapes:
        x = torch.randn(B, D, T, device=device, dtype=torch.bfloat16)
        weight = torch.randn(D, W, device=device, dtype=torch.bfloat16)
        bias = torch.randn(D, device=device, dtype=torch.bfloat16)
        opcheck(causal_conv1d_fwd, (x, weight, bias, "silu"), raise_exception=True)
        opcheck(causal_conv1d_fwd, (x, weight, None, "none"), raise_exception=True)


@requires_gpu
def test_causal_conv1d_bwd_opcheck() -> None:
    """Test causal_conv1d_bwd custom op with opcheck."""
    device = get_default_device()
    shapes = [
        (1, 32, 64, 2),
        (2, 64, 128, 4),
        (4, 128, 256, 3),
    ]
    for B, D, T, W in shapes:
        dout = torch.randn(B, D, T, device=device, dtype=torch.bfloat16)
        x = torch.randn(B, D, T, device=device, dtype=torch.bfloat16)
        weight = torch.randn(D, W, device=device, dtype=torch.bfloat16)
        bias = torch.randn(D, device=device, dtype=torch.bfloat16)
        opcheck(
            causal_conv1d_bwd,
            (dout, x, weight, bias, "silu"),
            raise_exception=True,
        )
        opcheck(
            causal_conv1d_bwd,
            (dout, x, weight, None, "none"),
            raise_exception=True,
        )


@requires_gpu
def test_causal_conv1d_compile() -> None:
    """Test that causal_conv1d works with torch.compile."""
    device = get_default_device()

    @torch.compile(fullgraph=True)
    def compiled_fn(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
        return causal_conv1d(x, weight, bias, activation=activation)

    x = torch.randn(2, 64, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, 4, device=device, dtype=torch.bfloat16)
    bias = torch.randn(64, device=device, dtype=torch.bfloat16)

    result = compiled_fn(x, weight, bias, "silu")
    expected = reference_causal_conv1d(x, weight, bias, activation="silu")

    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
