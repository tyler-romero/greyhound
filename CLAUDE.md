# CLAUDE.md

## Project Overview

Greyhound is a library of GPU kernels for PyTorch training and inference workloads.
Raw kernels are implemented as standalone CuTe DSL kernels and exposed to PyTorch through custom ops.

- **Language**: Python 3.11+
- **License**: Apache 2.0
- **Package manager**: `uv`
- **Package source root**: `src/`
- **GPU target**: CUDA

## Repository Structure

```text
src/
  greyhound/
    kernels/              # Raw standalone CuTe DSL kernels
      swiglu/
      rms_norm/
      chunked_linear_cross_entropy/
      causal_conv1d/
    ops/                  # torch.library.custom_op wrappers and autograd.Function classes
    nn/                   # Public functional API and nn.Module wrappers
    utils.py              # Shared runtime helpers
    testing.py            # GPU pytest marks
  tests/                  # pytest suite
  benchmarks/             # Benchmarks and comparison providers
scripts/                  # Local, Modal, Gantry, and release scripts
docs/                     # MkDocs documentation
```

The top-level `greyhound/` and `tests/` directories may contain stale local bytecode. The
packaged code and configured pytest suite live under `src/`.

## Development Commands

```bash
# Install the project and development tools
uv sync --extra dev

# Include optional third-party benchmark providers
uv sync --extra dev --extra thirdparty

# Run the CI checks
make checks
make test
mkdocs build --strict
python -m build

# Run one test file
pytest -v src/tests/nn/functional/swiglu_test.py
```

GPU tests are skipped automatically when CUDA is unavailable. A CPU-only run still verifies
collection and imports, but it does not validate kernel compilation or execution.

## Architecture

Greyhound uses three layers:

1. **`greyhound.kernels`** defines raw standalone CuTe DSL GPU kernels.
2. **`greyhound.ops`** exposes kernels as `torch.library.custom_op` operations, supplies fake
   implementations for tracing, and connects backward passes with `torch.autograd.Function`.
3. **`greyhound.nn`** provides public functional APIs and selected `nn.Module` wrappers.

### Implemented Kernels

| Family | Raw kernels | Public functional API | Module |
|--------|-------------|-----------------------|--------|
| SwiGLU | `swiglu_fwd_kernel`, `swiglu_bwd_kernel` | `swiglu` | `GreyhoundSwiGLUMLP` |
| RMSNorm | `rms_norm_fwd_kernel`, `rms_norm_bwd_kernel` | `rms_norm` | `GreyhoundRMSNorm` |
| Chunked Linear Cross-Entropy | `cross_entropy_with_grad_kernel` | `chunked_linear_cross_entropy` | - |
| Causal Conv1D | `causal_conv1d_fwd_kernel`, `causal_conv1d_bwd_kernel` | `causal_conv1d` | `GreyhoundCausalConv1d` |

Chunked Linear Cross-Entropy performs chunked linear algebra with PyTorch/cuBLAS and uses
a fused cross-entropy-and-logits-gradient CuTe DSL kernel per logits tile. Causal Conv1D,
RMSNorm, and SwiGLU use CuTe DSL kernels for forward and backward.

Prefer masked loads or in-kernel boundary checks over pre-padding tensors in the ops layer.
For causal Conv1D, avoiding `F.pad()` removes large temporary allocations and memory copies.

## Code Conventions

- Format with `ruff format .`; lint with `ruff check .`.
- Use double quotes, spaces, and a 100-character line limit.
- Run `ty check .` for type checking.
- Register custom ops under the `greyhound::` namespace.
- Put tests in `src/tests/` and name files `*_test.py`.
- Apply `@requires_gpu` from `greyhound.testing` to CUDA-dependent tests.
- Add focused correctness, gradient, `opcheck`, and `torch.compile` coverage where relevant.

## CI and Releases

GitHub Actions runs tests, Ruff linting, Ruff formatting checks, ty, package builds, and
`mkdocs build --strict` on Python 3.11. Pull requests must update `CHANGELOG.md`. Tags matching
`v*.*.*` trigger the PyPI and GitHub release workflow.

## Adding a Kernel

1. Add `src/greyhound/kernels/<name>/` with the raw CuTe DSL kernel implementation.
2. Add `src/greyhound/ops/<name>.py` with custom ops, fake implementations, and autograd
   integration as needed.
3. Export a public function from `src/greyhound/nn/functional.py`.
4. Add an `nn.Module` wrapper when there is a useful drop-in module API.
5. Add focused tests under `src/tests/`.
6. Add benchmarks and docs.
7. Update `CHANGELOG.md`.
