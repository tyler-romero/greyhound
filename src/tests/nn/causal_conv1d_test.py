import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from greyhound.nn.causal_conv1d import GreyhoundCausalConv1d
from greyhound.testing import requires_gpu
from greyhound.utils import get_default_device


def _reference_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference using nn.Conv1d-shaped weight [D, 1, W]."""
    D = x.shape[1]
    W = weight.shape[2]
    out = F.conv1d(x, weight, bias, padding=W - 1, groups=D)[..., : x.shape[-1]]
    if activation == "silu":
        return F.silu(out)
    return out


@requires_gpu
@pytest.mark.parametrize(
    ("channels", "kernel_size", "activation"),
    [
        (64, 4, None),
        (128, 2, "silu"),
        (256, 3, None),
    ],
)
def test_causal_conv1d_module_correctness(
    channels: int, kernel_size: int, activation: str | None
) -> None:
    """Test GreyhoundCausalConv1d matches reference depthwise conv1d."""
    device = get_default_device()

    module = GreyhoundCausalConv1d(
        channels=channels,
        kernel_size=kernel_size,
        bias=True,
        activation=activation,
        device=device,
        dtype=torch.bfloat16,
    )

    x = torch.randn(2, channels, 128, device=device, dtype=torch.bfloat16)
    result = module(x)
    expected = _reference_causal_conv1d(x, module.weight, module.bias, activation=activation)

    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), (
        f"Output mismatch for channels={channels}, kernel_size={kernel_size}, "
        f"activation={activation}"
    )


@requires_gpu
def test_causal_conv1d_module_no_bias() -> None:
    """Test GreyhoundCausalConv1d without bias."""
    device = get_default_device()

    module = GreyhoundCausalConv1d(
        channels=64,
        kernel_size=4,
        bias=False,
        activation="silu",
        device=device,
        dtype=torch.bfloat16,
    )

    assert module.bias is None
    x = torch.randn(2, 64, 128, device=device, dtype=torch.bfloat16)
    result = module(x)
    expected = _reference_causal_conv1d(x, module.weight, None, activation="silu")

    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


@requires_gpu
def test_causal_conv1d_module_gradient() -> None:
    """Test gradient flow through GreyhoundCausalConv1d."""
    device = get_default_device()

    module = GreyhoundCausalConv1d(
        channels=64,
        kernel_size=4,
        bias=True,
        activation="silu",
        device=device,
        dtype=torch.float32,
    )

    # Build a reference conv1d with the same weights
    ref = nn.Conv1d(64, 64, 4, groups=64, bias=True, device=device, dtype=torch.float32)
    with torch.no_grad():
        ref.weight.copy_(module.weight)
        ref.bias.copy_(module.bias)  # ty:ignore[unresolved-attribute, invalid-argument-type]

    x = torch.randn(2, 64, 128, device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    out = module(x)
    out_ref = _reference_causal_conv1d(x_ref, ref.weight, ref.bias, activation="silu")

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    assert x.grad is not None and x_ref.grad is not None
    assert torch.allclose(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4), "x.grad mismatch"

    assert module.weight.grad is not None and ref.weight.grad is not None
    # Module weight is [D, 1, W]; kernel returns grad for [D, W] which autograd reshapes
    assert torch.allclose(module.weight.grad, ref.weight.grad, rtol=1e-4, atol=1e-4), (
        "weight.grad mismatch"
    )

    assert module.bias.grad is not None and ref.bias.grad is not None  # ty:ignore[unresolved-attribute]
    assert torch.allclose(module.bias.grad, ref.bias.grad, rtol=1e-4, atol=1e-4), (  # ty:ignore[unresolved-attribute]
        "bias.grad mismatch"
    )


@requires_gpu
def test_causal_conv1d_module_is_conv1d() -> None:
    """Test that GreyhoundCausalConv1d is an instance of nn.Conv1d."""
    module = GreyhoundCausalConv1d(channels=64, kernel_size=4)
    assert isinstance(module, nn.Conv1d)
    assert module.in_channels == 64
    assert module.out_channels == 64
    assert module.kernel_size == (4,)
    assert module.groups == 64
    assert module.padding == (0,)
    assert module.stride == (1,)
    assert module.dilation == (1,)


@requires_gpu
def test_causal_conv1d_module_state_dict() -> None:
    """Test that state_dict is compatible with nn.Conv1d."""
    device = get_default_device()

    module = GreyhoundCausalConv1d(channels=64, kernel_size=4, device=device, dtype=torch.float32)
    state = module.state_dict()

    assert "weight" in state
    assert "bias" in state
    assert state["weight"].shape == (64, 1, 4)
    assert state["bias"].shape == (64,)

    # Load into a standard nn.Conv1d
    ref = nn.Conv1d(64, 64, 4, groups=64, device=device, dtype=torch.float32)
    ref.load_state_dict(state)

    assert torch.equal(ref.weight, module.weight)
    assert torch.equal(ref.bias, module.bias)  # ty:ignore[invalid-argument-type]
