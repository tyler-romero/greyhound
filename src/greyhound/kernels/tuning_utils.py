from typing import Callable

import torch


def do_bench(
    fn: Callable,
    warmup: int = 250,
    rep: int = 500,
    grad_to_none: list[torch.Tensor] | bool | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
) -> tuple:
    """Benchmark the runtime of the provided function using Triton's helper."""
    from triton.testing import do_bench  # type: ignore[import-untyped]

    return do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        grad_to_none=grad_to_none,
        quantiles=quantiles,
        return_mode=return_mode,
    )
