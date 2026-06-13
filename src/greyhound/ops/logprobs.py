from typing import Any

import torch
from torch import Tensor

from greyhound.kernels.logprobs import selective_log_softmax_kernel

__all__ = ["SelectiveLogSoftmaxFunction", "selective_log_softmax_fwd"]


@torch.library.custom_op("greyhound::selective_log_softmax_fwd", mutates_args=())
def selective_log_softmax_fwd(logits: Tensor, index: Tensor) -> Tensor:
    return selective_log_softmax_kernel(logits, index)


@selective_log_softmax_fwd.register_fake
def _(logits: Tensor, index: Tensor) -> Tensor:
    return torch.empty_like(index, dtype=logits.dtype, device=logits.device)


class SelectiveLogSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits: Tensor, index: Tensor) -> Tensor:
        del ctx
        return selective_log_softmax_fwd(logits, index)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, None]:  # ty:ignore[invalid-method-override]
        del ctx, grad_out
        raise NotImplementedError("selective_log_softmax does not implement backward yet")
