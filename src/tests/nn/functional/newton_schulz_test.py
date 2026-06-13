from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from greyhound.bonus.newton_schultz import DEFAULT_NS_CONSTS, orthogonalize_via_newton_schulz
from greyhound.testing import requires_symmetric_gemm
from greyhound.utils import get_default_device


def zeropower_via_newtonschulz5(
    g: Tensor,
    epsilon: float = 1e-7,
    ns_consts: Sequence[tuple[float, float, float]] = DEFAULT_NS_CONSTS,
) -> Tensor:
    x = g.to(dtype=torch.bfloat16)
    if g.size(-2) > g.size(-1):
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in ns_consts:
        gram = x @ x.mT
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    if g.size(-2) > g.size(-1):
        x = x.mT
    return x


@requires_symmetric_gemm
@pytest.mark.parametrize(
    "shape",
    [(16, 32), (32, 16), (64, 128), (2, 16, 24), (2, 24, 16)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_orthogonalize_via_newton_schulz_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    device = get_default_device()
    g = torch.randn(shape, device=device, dtype=dtype)

    result = orthogonalize_via_newton_schulz(g)
    expected = zeropower_via_newtonschulz5(g).to(dtype=result.dtype)

    assert result.dtype == g.dtype
    assert result.shape == g.shape
    torch.testing.assert_close(result, expected, rtol=2e-2, atol=3e-2)
