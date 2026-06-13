from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["ChunkedLinearLossFunction"]

_TARGET_LOGITS_CHUNK_BYTES = 512 * 1024 * 1024
_LARGE_D_MODEL_TARGET_LOGITS_CHUNK_BYTES = 2 * 1024 * 1024 * 1024


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _prev_power_of_2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _chunked_linear_loss_chunk_size(B: int, D: int, V: int, element_size: int) -> int:
    base = _next_power_of_2(_cdiv(B, max(_cdiv(V, D), 1)))
    base = min(max(base, 1), B)

    target_chunk_bytes = (
        _LARGE_D_MODEL_TARGET_LOGITS_CHUNK_BYTES if D >= 4096 else _TARGET_LOGITS_CHUNK_BYTES
    )
    bytes_per_row = max(V * element_size, 1)
    max_rows = max(target_chunk_bytes // bytes_per_row, 1)
    memory_limited = min(B, _prev_power_of_2(max_rows))
    return max(base, memory_limited)


def _resolve_chunk_size(
    B: int,
    D: int,
    V: int,
    element_size: int,
    chunk_size: int | None,
) -> int:
    if chunk_size is None:
        return _chunked_linear_loss_chunk_size(B, D, V, element_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return min(chunk_size, B)


def _check_scalar_loss(loss: Tensor) -> None:
    if loss.ndim != 0:
        raise ValueError(f"chunked loss function must return a scalar tensor, got {loss.shape}")


def _chunked_linear_loss_with_grad(
    inputs: Tensor,
    weight: Tensor,
    loss_and_grad_fn: Callable[[Tensor, int, int], tuple[Tensor, Tensor]],
    fp32_grad_weight_accum: bool,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Chunked linear projection with per-chunk loss and logits-gradient callback.

    ``weight`` is in transposed layout, [D, V]. The callback receives each logits
    chunk [C, V] plus its [start, end) row range and returns ``(loss, dlogits)``.
    The returned loss must be scalar and ``dlogits`` must have the same shape as
    the logits chunk.
    """
    B, D = inputs.shape
    V = weight.shape[1]
    chunk_size = _resolve_chunk_size(B, D, V, inputs.element_size(), chunk_size)

    loss_total = torch.zeros([], dtype=torch.float32, device=inputs.device)
    grad_input = torch.zeros_like(inputs)
    grad_weight_accum_dtype = torch.float32 if fp32_grad_weight_accum else weight.dtype
    grad_weight_accum = torch.zeros([D, V], dtype=grad_weight_accum_dtype, device=weight.device)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        input_chunk = inputs[start:end]  # [C, D]
        logits_chunk = input_chunk @ weight  # [C, V]
        loss_chunk, grad_logits = loss_and_grad_fn(logits_chunk, start, end)
        _check_scalar_loss(loss_chunk)
        if grad_logits.shape != logits_chunk.shape:
            raise ValueError(
                f"loss gradient shape must match logits shape {tuple(logits_chunk.shape)}, "
                f"got {tuple(grad_logits.shape)}"
            )
        loss_chunk = loss_chunk.detach()
        grad_logits = grad_logits.detach()

        loss_total = loss_total + loss_chunk.float()
        grad_input[start:end] = grad_logits @ weight.T
        torch.addmm(
            grad_weight_accum,
            input_chunk.T,
            grad_logits,
            out=grad_weight_accum,
            out_dtype=grad_weight_accum_dtype,
        )

    return loss_total, grad_input, grad_weight_accum.to(weight.dtype)


def _chunked_linear_loss_forward_only(
    inputs: Tensor,
    weight: Tensor,
    loss_and_grad_fn: Callable[[Tensor, int, int], tuple[Tensor, Tensor]],
    chunk_size: int | None = None,
) -> Tensor:
    """Chunked linear projection with a loss-and-logits-gradient callback."""
    B, D = inputs.shape
    V = weight.shape[1]
    chunk_size = _resolve_chunk_size(B, D, V, inputs.element_size(), chunk_size)

    loss_total = torch.zeros([], dtype=torch.float32, device=inputs.device)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        logits_chunk = inputs[start:end] @ weight
        loss_chunk, grad_logits = loss_and_grad_fn(logits_chunk, start, end)
        _check_scalar_loss(loss_chunk)
        if grad_logits.shape != logits_chunk.shape:
            raise ValueError(
                f"loss gradient shape must match logits shape {tuple(logits_chunk.shape)}, "
                f"got {tuple(grad_logits.shape)}"
            )
        loss_total = loss_total + loss_chunk.detach().float()
    return loss_total


class ChunkedLinearLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,  # [B, D]
        weight: torch.Tensor,  # [D, V]
        loss_and_grad_fn: Callable[[Tensor, int, int], tuple[Tensor, Tensor]],
        chunk_size: int | None = None,
        fp32_grad_weight_accum: bool = True,
    ) -> torch.Tensor:
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:  # ty:ignore[unresolved-attribute]
            return _chunked_linear_loss_forward_only(inputs, weight, loss_and_grad_fn, chunk_size)

        loss, grad_input, grad_weight = _chunked_linear_loss_with_grad(
            inputs,
            weight,
            loss_and_grad_fn,
            fp32_grad_weight_accum,
            chunk_size,
        )
        ctx.save_for_backward(grad_input, grad_weight)  # ty:ignore[unresolved-attribute]
        return loss

    @staticmethod
    def backward(
        ctx: object, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:  # ty:ignore[invalid-method-override]
        grad_input, grad_weight = ctx.saved_tensors  # ty:ignore[unresolved-attribute]
        return (
            grad_input * grad_output,
            grad_weight * grad_output,
            None,  # loss_and_grad_fn
            None,  # chunk_size
            None,  # fp32_grad_weight_accum
        )
