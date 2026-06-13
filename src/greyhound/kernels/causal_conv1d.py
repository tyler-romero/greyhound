from functools import lru_cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch import Tensor

__all__ = ["causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"]

_THREADS = 256
_ITEMS_PER_THREAD = 16
_WARP_SIZE = 32


@cute.kernel
def _causal_conv1d_fwd_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    channels: cutlass.Constexpr,
    seqlen: cutlass.Constexpr,
    width: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    silu: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tiles_per_channel = cute.ceil_div(seqlen, _THREADS * _ITEMS_PER_THREAD)
    channel_idx = bidx // tiles_per_channel
    d = channel_idx % channels
    b = channel_idx // channels
    tile_idx = bidx % tiles_per_channel

    for item in cutlass.range_constexpr(_ITEMS_PER_THREAD):
        t = tile_idx * (_THREADS * _ITEMS_PER_THREAD) + tidx + item * _THREADS
        if t < seqlen:
            acc = cutlass.Float32(0.0)
            for wi in cutlass.range_constexpr(width):
                if t + wi >= width - 1:
                    x_val = x[b, d, t + wi - (width - 1)].to(cutlass.Float32)
                    w_val = weight[d, wi].to(cutlass.Float32)
                    acc += x_val * w_val
            if has_bias:
                acc += bias[d].to(cutlass.Float32)
            if silu:
                acc = acc / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=True))
            out[b, d, t] = acc.to(out.element_type)


@cute.jit
def _launch_causal_conv1d_fwd(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
    channels: cutlass.Constexpr,
    seqlen: cutlass.Constexpr,
    width: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    silu: cutlass.Constexpr,
):
    grid = cute.size(x, mode=[0]) * channels * cute.ceil_div(seqlen, _THREADS * _ITEMS_PER_THREAD)
    _causal_conv1d_fwd_kernel(x, weight, bias, out, channels, seqlen, width, has_bias, silu).launch(
        grid=[grid, 1, 1], block=[_THREADS, 1, 1]
    )


@cute.kernel
def _causal_conv1d_bwd_kernel(
    dout: cute.Tensor,
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    dx: cute.Tensor,
    dweight: cute.Tensor,
    dbias: cute.Tensor,
    channels: cutlass.Constexpr,
    seqlen: cutlass.Constexpr,
    width: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    silu: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tiles_per_channel = cute.ceil_div(seqlen, _THREADS * _ITEMS_PER_THREAD)
    channel_idx = bidx // tiles_per_channel
    d = channel_idx % channels
    b = channel_idx // channels
    tile_idx = bidx % tiles_per_channel
    lane_idx = tidx % _WARP_SIZE

    dw = [
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
    ]
    db = cutlass.Float32(0.0)
    for item in cutlass.range_constexpr(_ITEMS_PER_THREAD):
        t = tile_idx * (_THREADS * _ITEMS_PER_THREAD) + tidx + item * _THREADS
        if t < seqlen:
            dout_act = dout[b, d, t].to(cutlass.Float32)
            if silu:
                pre_act = cutlass.Float32(0.0)
                for wi in cutlass.range_constexpr(width):
                    if t + wi >= width - 1:
                        x_val = x[b, d, t + wi - (width - 1)].to(cutlass.Float32)
                        pre_act += x_val * weight[d, wi].to(cutlass.Float32)
                if has_bias:
                    pre_act += bias[d].to(cutlass.Float32)
                sig = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-pre_act, fastmath=True)
                )
                dout_act *= sig * (cutlass.Float32(1.0) + pre_act * (cutlass.Float32(1.0) - sig))

            db += dout_act
            for wi in cutlass.range_constexpr(width):
                if t + wi >= width - 1:
                    dw[wi] += dout_act * x[b, d, t + wi - (width - 1)].to(cutlass.Float32)

            dx_val = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(width):
                shifted_t = t + j
                if shifted_t < seqlen:
                    dout_shifted = dout[b, d, shifted_t].to(cutlass.Float32)
                    if silu:
                        pre_shifted = cutlass.Float32(0.0)
                        for wi in cutlass.range_constexpr(width):
                            if shifted_t + wi >= width - 1:
                                x_val = x[b, d, shifted_t + wi - (width - 1)].to(cutlass.Float32)
                                pre_shifted += x_val * weight[d, wi].to(cutlass.Float32)
                        if has_bias:
                            pre_shifted += bias[d].to(cutlass.Float32)
                        sig = cutlass.Float32(1.0) / (
                            cutlass.Float32(1.0) + cute.exp(-pre_shifted, fastmath=True)
                        )
                        dout_shifted *= sig * (
                            cutlass.Float32(1.0) + pre_shifted * (cutlass.Float32(1.0) - sig)
                        )
                    dx_val += weight[d, width - 1 - j].to(cutlass.Float32) * dout_shifted
            dx[b, d, t] = dx_val.to(dx.element_type)

    for offset in (16, 8, 4, 2, 1):
        for wi in cutlass.range_constexpr(width):
            dw[wi] += cute.arch.shuffle_sync_down(dw[wi], offset)
        db += cute.arch.shuffle_sync_down(db, offset)
    if lane_idx == 0:
        for wi in cutlass.range_constexpr(width):
            cute.arch.atomic_add(dweight.iterator + d * width + wi, dw[wi])
        if has_bias:
            cute.arch.atomic_add(dbias.iterator + d, db)


@cute.jit
def _launch_causal_conv1d_bwd(
    dout: cute.Tensor,
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    dx: cute.Tensor,
    dweight: cute.Tensor,
    dbias: cute.Tensor,
    channels: cutlass.Constexpr,
    seqlen: cutlass.Constexpr,
    width: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    silu: cutlass.Constexpr,
):
    grid = cute.size(x, mode=[0]) * channels * cute.ceil_div(seqlen, _THREADS * _ITEMS_PER_THREAD)
    _causal_conv1d_bwd_kernel(
        dout, x, weight, bias, dx, dweight, dbias, channels, seqlen, width, has_bias, silu
    ).launch(grid=[grid, 1, 1], block=[_THREADS, 1, 1])


def _as_cute_tensor(tensor: Tensor) -> Any:
    return from_dlpack(tensor.detach()).mark_layout_dynamic(leading_dim=-1)


def _validate_inputs(
    x: Tensor, weight: Tensor, bias: Tensor | None, activation: str | None
) -> None:
    if not x.is_cuda or not weight.is_cuda or (bias is not None and not bias.is_cuda):
        raise ValueError("CuTe DSL causal_conv1d requires CUDA tensors")
    if weight.device != x.device or (bias is not None and bias.device != x.device):
        raise ValueError("x, weight, and bias must be on the same device")
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError("Expected x[B, D, T] and weight[D, W]")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("x and weight channel dimensions must match")
    if weight.shape[1] not in (2, 3, 4):
        raise ValueError("CuTe DSL shortconv supports widths 2, 3, and 4")
    if bias is not None and bias.shape != (x.shape[1],):
        raise ValueError("bias must have shape [D]")
    if activation not in (None, "silu", "swish"):
        raise ValueError("activation must be None, 'silu', or 'swish'")
    if (
        not x.is_contiguous()
        or not weight.is_contiguous()
        or (bias is not None and not bias.is_contiguous())
    ):
        raise ValueError("CuTe DSL causal_conv1d requires contiguous tensors")
    if x.dtype != weight.dtype or (bias is not None and x.dtype != bias.dtype):
        raise ValueError("x, weight, and bias must have the same dtype")


@lru_cache(maxsize=None)
def _compile(
    batch_size: int,
    channels: int,
    seqlen: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    has_bias: bool,
    silu: bool,
) -> Any:
    x = torch.empty((batch_size, channels, seqlen), device=device, dtype=dtype)
    weight = torch.empty((channels, width), device=device, dtype=dtype)
    bias = torch.empty((channels,), device=device, dtype=dtype)
    out = torch.empty_like(x)
    args = [_as_cute_tensor(tensor) for tensor in (x, weight, bias, out)]
    return cute.compile(_launch_causal_conv1d_fwd, *args, channels, seqlen, width, has_bias, silu)


@lru_cache(maxsize=None)
def _compile_bwd(
    batch_size: int,
    channels: int,
    seqlen: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    has_bias: bool,
    silu: bool,
) -> Any:
    dout = torch.empty((batch_size, channels, seqlen), device=device, dtype=dtype)
    x = torch.empty_like(dout)
    weight = torch.empty((channels, width), device=device, dtype=dtype)
    bias = torch.empty((channels,), device=device, dtype=dtype)
    dx = torch.empty_like(x)
    dweight = torch.empty((channels, width), device=device, dtype=torch.float32)
    dbias = torch.empty((channels,), device=device, dtype=torch.float32)
    args = [_as_cute_tensor(tensor) for tensor in (dout, x, weight, bias, dx, dweight, dbias)]
    return cute.compile(_launch_causal_conv1d_bwd, *args, channels, seqlen, width, has_bias, silu)


def causal_conv1d_fwd_kernel(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: bool = False,
    has_bias: bool = False,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Run the standalone CuTe DSL short causal depthwise convolution forward kernel."""
    if has_bias != (bias is not None):
        raise ValueError("has_bias must match whether bias is provided")
    _validate_inputs(x, weight, bias, "silu" if activation else None)
    batch_size, channels, seqlen = x.shape
    width = weight.shape[1]
    has_bias = bias is not None
    if out is None:
        out = torch.empty_like(x)
    elif (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous CUDA tensor matching x")

    bias_arg = (
        bias if bias is not None else torch.empty((channels,), device=x.device, dtype=x.dtype)
    )
    compiled = _compile(
        batch_size, channels, seqlen, width, x.dtype, x.device, has_bias, activation
    )
    args = [_as_cute_tensor(tensor) for tensor in (x, weight, bias_arg, out)]
    compiled(*args)
    return out


def causal_conv1d_bwd_kernel(
    dout: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: bool = False,
    has_bias: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the standalone CuTe DSL short causal depthwise convolution backward kernel."""
    if has_bias != (bias is not None):
        raise ValueError("has_bias must match whether bias is provided")
    _validate_inputs(x, weight, bias, "silu" if activation else None)
    if dout.shape != x.shape or dout.dtype != x.dtype or dout.device != x.device:
        raise ValueError("dout must match x")
    if not dout.is_contiguous():
        raise ValueError("CuTe DSL causal_conv1d requires contiguous dout")

    batch_size, channels, seqlen = x.shape
    width = weight.shape[1]
    has_bias = bias is not None
    bias_arg = (
        bias if bias is not None else torch.empty((channels,), device=x.device, dtype=x.dtype)
    )
    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    dbias = torch.zeros((channels,), device=x.device, dtype=torch.float32)
    compiled = _compile_bwd(
        batch_size, channels, seqlen, width, x.dtype, x.device, has_bias, activation
    )
    args = [_as_cute_tensor(tensor) for tensor in (dout, x, weight, bias_arg, dx, dweight, dbias)]
    compiled(*args)
    return dx, dweight.to(weight.dtype), dbias.to(bias.dtype if bias is not None else weight.dtype)
