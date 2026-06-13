from typing import Literal

import torch

try:
    from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d

    has_fla = True
except ImportError:
    has_fla = False

__all__ = ["fla_causal_conv1d"]


def fla_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    activation: Literal["silu", "swish"] | None = None,
) -> torch.Tensor:
    """Wrap FLA's causal_conv1d to accept [B, D, T] layout."""
    assert has_fla, "FLA is not installed"
    out, _final_state = _fla_causal_conv1d(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        residual=residual.transpose(1, 2) if residual is not None else None,
        activation=activation,
        backend="triton",  # use fla's Triton backend
    )
    return out.transpose(1, 2)
