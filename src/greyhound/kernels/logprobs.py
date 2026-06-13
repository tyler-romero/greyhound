from functools import lru_cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch import Tensor

__all__ = ["selective_log_softmax_kernel"]

_THREADS = 1024
_WARP_SIZE = 32
_WARPS = _THREADS // _WARP_SIZE
_NEG_INF = -3.4028234663852886e38
_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453


@cute.kernel
def _selective_log_softmax_kernel(
    logits: cute.Tensor,
    index: cute.Tensor,
    out: cute.Tensor,
    vocab_size: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane_idx = tidx % _WARP_SIZE
    warp_idx = tidx // _WARP_SIZE
    partials = cute.arch.alloc_smem(cutlass.Float32, _WARPS * 2)

    local_max = cutlass.Float32(_NEG_INF)
    local_sum = cutlass.Float32(0.0)
    for col in cutlass.range(tidx, vocab_size, _THREADS):
        x = logits[row, col].to(cutlass.Float32)
        new_max = cute.arch.fmax(local_max, x)
        local_sum = local_sum * cute.exp2(
            (local_max - new_max) * _LOG2_E, fastmath=True
        ) + cute.exp2(
            (x - new_max) * _LOG2_E,
            fastmath=True,
        )
        local_max = new_max

    for offset in (16, 8, 4, 2, 1):
        other_max = cute.arch.shuffle_sync_down(local_max, offset)
        other_sum = cute.arch.shuffle_sync_down(local_sum, offset)
        new_max = cute.arch.fmax(local_max, other_max)
        if local_sum == 0.0 and other_sum == 0.0:
            local_sum = cutlass.Float32(0.0)
        else:
            local_sum = local_sum * cute.exp2(
                (local_max - new_max) * _LOG2_E, fastmath=True
            ) + other_sum * cute.exp2((other_max - new_max) * _LOG2_E, fastmath=True)
        local_max = new_max
    if lane_idx == 0:
        partials[warp_idx] = local_max
        partials[_WARPS + warp_idx] = local_sum
    cute.arch.sync_threads()

    if warp_idx == 0:
        block_max = cutlass.Float32(_NEG_INF)
        block_sum = cutlass.Float32(0.0)
        if lane_idx < _WARPS:
            block_max = partials[lane_idx]
            block_sum = partials[_WARPS + lane_idx]
        for offset in (16, 8, 4, 2, 1):
            other_max = cute.arch.shuffle_sync_down(block_max, offset)
            other_sum = cute.arch.shuffle_sync_down(block_sum, offset)
            new_max = cute.arch.fmax(block_max, other_max)
            if block_sum == 0.0 and other_sum == 0.0:
                block_sum = cutlass.Float32(0.0)
            else:
                block_sum = block_sum * cute.exp2(
                    (block_max - new_max) * _LOG2_E, fastmath=True
                ) + other_sum * cute.exp2((other_max - new_max) * _LOG2_E, fastmath=True)
            block_max = new_max
        if lane_idx == 0:
            partials[0] = block_max
            partials[_WARPS] = block_sum
    cute.arch.sync_threads()

    if tidx == 0:
        target = index[row]
        lse = partials[0] + cute.log2(partials[_WARPS], fastmath=True) * _LN_2
        selected = logits[row, target].to(cutlass.Float32)
        out[row] = (selected - lse).to(out.element_type)


@cute.jit
def _launch_selective_log_softmax(
    logits: cute.Tensor,
    index: cute.Tensor,
    out: cute.Tensor,
    vocab_size: cutlass.Constexpr,
):
    _selective_log_softmax_kernel(logits, index, out, vocab_size).launch(
        grid=[cute.size(logits, mode=[0]), 1, 1],
        block=[_THREADS, 1, 1],
    )


def _as_cute_tensor(tensor: Tensor) -> Any:
    return from_dlpack(tensor.detach()).mark_layout_dynamic(leading_dim=-1)


@lru_cache(maxsize=None)
def _compile_selective_log_softmax(
    vocab_size: int,
    logits_dtype: torch.dtype,
    index_dtype: torch.dtype,
    device: torch.device,
) -> Any:
    # The row dimension is runtime-dynamic after mark_layout_dynamic(), while
    # vocab_size is the only shape value baked into the kernel. Keep compile
    # representatives tiny so near-capacity benchmark inputs do not require a
    # second full logits allocation just to compile.
    logits = torch.empty((1, vocab_size), dtype=logits_dtype, device=device)
    index = torch.empty((1,), dtype=index_dtype, device=device)
    out = torch.empty((1,), dtype=logits_dtype, device=device)
    return cute.compile(
        _launch_selective_log_softmax,
        _as_cute_tensor(logits),
        _as_cute_tensor(index),
        _as_cute_tensor(out),
        vocab_size,
    )


def selective_log_softmax_kernel(logits: Tensor, index: Tensor) -> Tensor:
    """Compute ``gather(log_softmax(logits, -1), index)`` without materializing logprobs.

    ``logits`` is flattened over every leading dimension and reduced row-by-row
    across the final vocabulary dimension. The reduction uses fp32 online
    log-sum-exp accumulation and writes one selected log-probability per row.
    """
    if logits.ndim < 2:
        raise ValueError(f"expected logits with at least 2 dimensions, got {logits.ndim}")
    if index.shape != logits.shape[:-1]:
        raise ValueError(
            f"expected index shape {tuple(logits.shape[:-1])}, got {tuple(index.shape)}"
        )
    if not logits.is_cuda or index.device != logits.device:
        raise ValueError("expected logits and index on the same CUDA device")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported logits dtype: {logits.dtype}")
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"expected int32 or int64 index, got {index.dtype}")
    if logits.shape[-1] <= 0:
        raise ValueError("expected non-empty vocabulary dimension")
    if not logits.is_contiguous() or not index.is_contiguous():
        raise ValueError("expected contiguous logits and index")

    vocab_size = logits.shape[-1]
    logits_2d = logits.view(-1, vocab_size)
    index_1d = index.view(-1)
    out = torch.empty_like(index, dtype=logits.dtype)
    out_1d = out.view(-1)

    compiled = _compile_selective_log_softmax(
        vocab_size,
        logits.dtype,
        index.dtype,
        logits.device,
    )
    compiled(
        _as_cute_tensor(logits_2d),
        _as_cute_tensor(index_1d),
        _as_cute_tensor(out_1d),
    )
    return out
