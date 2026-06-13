from typing import Any

import torch
from torch import Tensor

from greyhound.kernels.causal_conv1d import (
    causal_conv1d_bwd_kernel,
    causal_conv1d_fwd_kernel,
)

__all__ = ["CausalConv1dFunction", "causal_conv1d_fwd", "causal_conv1d_bwd"]


@torch.library.custom_op("greyhound::causal_conv1d_fwd", mutates_args=())
def causal_conv1d_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation_str: str,
) -> Tensor:
    has_bias = bias is not None
    activation = activation_str == "silu"
    return causal_conv1d_fwd_kernel(x, weight, bias, activation, has_bias)


@causal_conv1d_fwd.register_fake
def _(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation_str: str,
) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("greyhound::causal_conv1d_bwd", mutates_args=())
def causal_conv1d_bwd(
    dout: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation_str: str,
) -> tuple[Tensor, Tensor, Tensor]:
    activation = activation_str == "silu"
    has_bias = bias is not None

    return causal_conv1d_bwd_kernel(dout, x, weight, bias, activation, has_bias)


@causal_conv1d_bwd.register_fake
def _(
    dout: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    activation_str: str,
) -> tuple[Tensor, Tensor, Tensor]:
    D, W = weight.shape
    return (
        torch.empty_like(dout),
        weight.new_empty([D, W]),
        weight.new_empty([D]),
    )


class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        activation_str: str,
    ) -> Tensor:
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.activation_str = activation_str
        ctx.has_bias = bias is not None
        return causal_conv1d_fwd(x, weight, bias, activation_str)

    @staticmethod
    def backward(ctx: Any, dout: Tensor) -> tuple[Tensor, Tensor, Tensor | None, None]:  # ty:ignore[invalid-method-override]
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            bias = None
        dx, dweight, dbias = causal_conv1d_bwd(
            dout.contiguous(), x, weight, bias, ctx.activation_str
        )
        return dx, dweight, dbias if ctx.has_bias else None, None
