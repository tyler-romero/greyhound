from functools import lru_cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch import Tensor

__all__ = ["cross_entropy_with_grad_kernel"]

_THREADS = 1024
_WARP_SIZE = 32
_WARPS = _THREADS // _WARP_SIZE
_NEG_INF = -3.4028234663852886e38
_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453


@cute.kernel
def _cross_entropy_with_grad_kernel(
    logits: cute.Tensor,
    labels: cute.Tensor,
    ce_loss_sum: cute.Tensor,
    z_loss_sum: cute.Tensor,
    n_valid: cute.Tensor,
    vocab_size: cutlass.Constexpr,
    z_loss_multiplier: cutlass.Constexpr,
    ignore_index: cutlass.Constexpr,
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
            block_sum = block_sum * cute.exp2(
                (block_max - new_max) * _LOG2_E, fastmath=True
            ) + other_sum * cute.exp2((other_max - new_max) * _LOG2_E, fastmath=True)
            block_max = new_max
        if lane_idx == 0:
            partials[0] = block_max
            partials[_WARPS] = block_sum
    cute.arch.sync_threads()
    block_max = partials[0]
    block_sum = partials[_WARPS]

    lse = block_max + cute.log2(block_sum, fastmath=True) * _LN_2
    label = labels[row]
    is_valid = label != ignore_index
    if tidx == 0:
        if is_valid:
            ce_loss = lse - logits[row, label].to(cutlass.Float32)
            cute.arch.atomic_add(ce_loss_sum.iterator, ce_loss)
            cute.arch.atomic_add(z_loss_sum.iterator, lse * lse)
            cute.arch.atomic_add(n_valid.iterator, cutlass.Float32(1.0))

    z_grad_factor = cutlass.Float32(1.0) + z_loss_multiplier * cutlass.Float32(2.0) * lse
    for col in cutlass.range(tidx, vocab_size, _THREADS):
        grad = (
            cute.exp2(
                (logits[row, col].to(cutlass.Float32) - block_max) * _LOG2_E,
                fastmath=True,
            )
            / block_sum
            * z_grad_factor
        )
        if col == label:
            grad -= cutlass.Float32(1.0)
        if not is_valid:
            grad = cutlass.Float32(0.0)
        logits[row, col] = grad.to(logits.element_type)


@cute.jit
def _launch_cross_entropy_with_grad(
    logits: cute.Tensor,
    labels: cute.Tensor,
    ce_loss_sum: cute.Tensor,
    z_loss_sum: cute.Tensor,
    n_valid: cute.Tensor,
    vocab_size: cutlass.Constexpr,
    z_loss_multiplier: cutlass.Constexpr,
    ignore_index: cutlass.Constexpr,
):
    _cross_entropy_with_grad_kernel(
        logits,
        labels,
        ce_loss_sum,
        z_loss_sum,
        n_valid,
        vocab_size,
        z_loss_multiplier,
        ignore_index,
    ).launch(grid=[cute.size(logits, mode=[0]), 1, 1], block=[_THREADS, 1, 1])


def _as_cute_tensor(tensor: Tensor) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor.detach())
    if tensor.ndim > 0:
        return cute_tensor.mark_layout_dynamic(leading_dim=-1)
    return cute_tensor


@lru_cache(maxsize=None)
def _compile_cross_entropy_with_grad(
    rows: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    z_loss_multiplier: float,
    ignore_index: int,
) -> Any:
    logits = torch.empty((rows, vocab_size), dtype=dtype, device=device)
    labels = torch.empty((rows,), dtype=torch.int64, device=device)
    scalar = torch.empty((), dtype=torch.float32, device=device)
    return cute.compile(
        _launch_cross_entropy_with_grad,
        _as_cute_tensor(logits),
        _as_cute_tensor(labels),
        _as_cute_tensor(scalar),
        _as_cute_tensor(scalar),
        _as_cute_tensor(scalar),
        vocab_size,
        z_loss_multiplier,
        ignore_index,
    )


def cross_entropy_with_grad_kernel(
    logits: Tensor,
    labels: Tensor,
    z_loss_multiplier: float,
    ignore_index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute summed losses and overwrite ``logits`` with its gradient."""
    if logits.ndim != 2:
        raise ValueError(f"expected 2D logits, got shape {tuple(logits.shape)}")
    if labels.shape != (logits.shape[0],):
        raise ValueError(f"expected labels shape {(logits.shape[0],)}, got {tuple(labels.shape)}")
    if not logits.is_cuda or labels.device != logits.device:
        raise ValueError("expected logits and labels on the same CUDA device")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported logits dtype: {logits.dtype}")
    if labels.dtype != torch.int64:
        raise TypeError(f"expected int64 labels, got {labels.dtype}")
    if not logits.is_contiguous() or not labels.is_contiguous():
        raise ValueError("expected contiguous logits and labels")

    ce_loss_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
    z_loss_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
    n_valid = torch.zeros((), dtype=torch.float32, device=logits.device)
    compiled = _compile_cross_entropy_with_grad(
        logits.shape[0],
        logits.shape[1],
        logits.dtype,
        logits.device,
        float(z_loss_multiplier),
        int(ignore_index),
    )
    compiled(
        _as_cute_tensor(logits),
        _as_cute_tensor(labels),
        _as_cute_tensor(ce_loss_sum),
        _as_cute_tensor(z_loss_sum),
        _as_cute_tensor(n_valid),
    )
    return ce_loss_sum, z_loss_sum, n_valid
