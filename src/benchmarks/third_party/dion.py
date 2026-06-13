from torch import Tensor

try:
    from dion.newton_schulz_triton import newton_schulz_triton as _dion_newton_schulz

    has_dion = True
except Exception:
    has_dion = False


def dion_newton_schulz(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    assert has_dion, "Dion is not installed"
    return _dion_newton_schulz(G, epsilon)
