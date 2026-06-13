from typing import Sequence

import torch
from torch import Tensor

try:
    from quack.gemm_interface import gemm_symmetric as quack_gemm_symmetric

    has_quack = True
except ImportError:
    has_quack = False

DEFAULT_NS_CONSTS: tuple[tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def has_quack_symmetric_gemm(device: torch.device | str | None = None) -> bool:
    """Return whether Quack symmetric GEMM is expected to run on this device."""
    if not has_quack or not torch.cuda.is_available():
        return False
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        return False
    device_index = torch.cuda.current_device() if device.index is None else device.index
    capability = torch.cuda.get_device_capability(device_index)
    # Quack's SM80 symmetric GEMM entry point currently raises NotImplementedError.
    return capability[0] >= 9


@torch.no_grad
@torch.compile
def _cast_transpose_and_norm(g: Tensor, epsilon: float, transposed: bool) -> Tensor:
    x = g.bfloat16()
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)
    x = x.contiguous()
    return x


@torch.no_grad
def orthogonalize_via_newton_schulz(
    g: Tensor,
    epsilon: float = 1e-7,
    ns_consts: Sequence[tuple[float, float, float]] | None = None,
) -> Tensor:
    """
    Newton-Schulz zeropower iteration.

    Computes the quintic Newton-Schulz orthogonalization used by Muon-style
    optimizers. Inputs are cast to bfloat16 for the iteration, normalized by their
    Frobenius norm, and tall matrices are transposed internally so the symmetric
    matrix products use the smaller dimension.

    Uses Quack's symmetric GEMM for the Gram matrix computations to save flops.

    Args:
        g: Input tensor of shape ``[M, N]`` or ``[B, M, N]``.
        epsilon: Small constant added to the input norm for numerical stability.
        ns_consts: Optional sequence of ``(a, b, c)`` polynomial coefficients.

    Returns:
        Tensor with the same shape as ``g`` and dtype ``torch.bfloat16``.
    """
    if quack_gemm_symmetric is None or not has_quack_symmetric_gemm(g.device):
        raise RuntimeError("Quack symmetric GEMM is not available on this GPU")
    if ns_consts is None:
        ns_consts = DEFAULT_NS_CONSTS

    transposed = g.size(-2) > g.size(-1)
    x = _cast_transpose_and_norm(g, epsilon, transposed)

    rows = x.size(-2)
    gram = torch.empty((*x.shape[:-1], rows), dtype=x.dtype, device=x.device)
    update = torch.empty_like(gram)
    out = torch.empty_like(x)

    for a, b, c in ns_consts:
        quack_gemm_symmetric(x, x.mT, out=gram)
        quack_gemm_symmetric(gram, gram.mT, C=gram, out=update, alpha=c, beta=b)
        if x.ndim == 3:
            torch.baddbmm(x, update, x, beta=a, out=out)
        else:
            torch.addmm(x, update, x, beta=a, out=out)
        x, out = out, x

    if transposed:
        x = x.mT
    return x.to(dtype=g.dtype)
