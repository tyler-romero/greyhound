from functools import lru_cache

from torch import Tensor

try:
    from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

    has_gram_newton_schulz = True
except Exception:
    has_gram_newton_schulz = False


@lru_cache(maxsize=None)
def _get_gram_newton_schulz(epsilon: float) -> GramNewtonSchulz:
    assert has_gram_newton_schulz, "gram-newton-schulz is not installed"
    return GramNewtonSchulz(
        ns_epsilon=epsilon,
        ns_coefficients=[list(coefficients) for coefficients in POLAR_EXPRESS_COEFFICIENTS],
        gram_newton_schulz_reset_iterations=[2],
    )


def gram_newton_schulz(g: Tensor, epsilon: float = 1e-7) -> Tensor:
    return _get_gram_newton_schulz(epsilon)(g)
