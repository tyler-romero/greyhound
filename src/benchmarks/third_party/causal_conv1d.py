import torch

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn

    has_causal_conv1d = True
except ImportError:
    has_causal_conv1d = False


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    assert _causal_conv1d_fn is not None, "causal_conv1d is not installed"
    return _causal_conv1d_fn(x=x, weight=weight, bias=bias, activation=activation)
