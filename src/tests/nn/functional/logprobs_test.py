import pytest
import torch
from torch import Tensor
from torch.library import opcheck

from greyhound.nn.functional import selective_log_softmax
from greyhound.ops.logprobs import selective_log_softmax_fwd
from greyhound.testing import requires_gpu
from greyhound.utils import get_default_device


def reference_selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@requires_gpu
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.bfloat16, id="bfloat16"),
        pytest.param(torch.float32, id="float32"),
    ],
)
def test_selective_log_softmax_correctness(dtype: torch.dtype) -> None:
    device = get_default_device()
    logits = torch.randn(4, 16, 257, device=device, dtype=dtype)
    index = torch.randint(0, logits.shape[-1], logits.shape[:-1], device=device)

    result = selective_log_softmax(logits, index)
    expected = reference_selective_log_softmax(logits, index)

    assert result.shape == index.shape
    assert torch.allclose(result.float(), expected.float(), rtol=5e-3, atol=5e-3)


@requires_gpu
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_selective_log_softmax_index_dtypes(index_dtype: torch.dtype) -> None:
    device = get_default_device()
    logits = torch.randn(8, 513, device=device, dtype=torch.bfloat16)
    index = torch.randint(0, logits.shape[-1], logits.shape[:-1], device=device, dtype=index_dtype)

    result = selective_log_softmax(logits, index)
    expected = reference_selective_log_softmax(logits, index.long())

    assert torch.allclose(result.float(), expected.float(), rtol=5e-3, atol=5e-3)


@requires_gpu
def test_selective_log_softmax_fwd_opcheck() -> None:
    device = get_default_device()
    logits = torch.randn(4, 16, 256, device=device, dtype=torch.bfloat16)
    index = torch.randint(0, logits.shape[-1], logits.shape[:-1], device=device)

    opcheck(selective_log_softmax_fwd, (logits, index), raise_exception=True)
