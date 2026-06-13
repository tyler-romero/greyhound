import pytest
import torch

has_cuda = torch.cuda.is_available()
GPU_MARKS = (pytest.mark.gpu, pytest.mark.skipif(not has_cuda, reason="Requires a GPU"))


def requires_gpu(func):
    for mark in GPU_MARKS:
        func = mark(func)
    return func


def has_symmetric_gemm() -> bool:
    from greyhound.bonus.newton_schultz import has_quack_symmetric_gemm

    return has_quack_symmetric_gemm()


def requires_symmetric_gemm(func):
    func = requires_gpu(func)
    return pytest.mark.skipif(
        not has_symmetric_gemm(),
        reason="Requires Quack symmetric GEMM on this GPU",
    )(func)
