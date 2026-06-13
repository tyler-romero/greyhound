---
title: Benchmarking
description: How greyhound benchmarks kernels, records results, and produces documentation plots.
---

Greyhound benchmarks compare the fused kernels against PyTorch eager,
`torch.compile`, and available third-party kernels. The scripts live under
`src/benchmarks/` and write normalized result rows to
`src/benchmarks/data/benchmark_data.csv`.

Benchmark rows include the kernel name, provider, operation mode, metric, p50/p20/p80
values, extra configuration, GPU name, and git commit. Plot generation reads that CSV
and produces the static and interactive figures used throughout the docs.

## Running Benchmarks

Run benchmark scripts from the repository root with `uv run python`:

```bash
uv run python src/benchmarks/chunked_linear_cross_entropy_bench.py --mode full
```

Each benchmark accepts provider, shape, dtype, and mode options. Most kernels support
the same operation modes:

- `full`: forward + backward, usually the most training-relevant measurement
- `forward`: forward pass only
- `backward`: backward pass only, when the operation has a separate backward path

For example:

```bash
uv run python src/benchmarks/chunked_linear_cross_entropy_bench.py \
  --mode full \
  --providers greyhound,torch-eager,torch-compile \
  --batch-size 8192 \
  --d-models 4096,8192 \
  --vocab-sizes 65536,128256
```

Use `--help` on any benchmark script to see its shape sweep and provider options:

```bash
uv run python src/benchmarks/causal_conv1d_bench.py --help
```

## Benchmark Scripts

The main benchmark entry points are:

- `src/benchmarks/causal_conv1d_bench.py`
- `src/benchmarks/cross_entropy_with_grad_bench.py`
- `src/benchmarks/chunked_linear_cross_entropy_bench.py`
- `src/benchmarks/logprobs_bench.py`
- `src/benchmarks/newton_schulz_bench.py`

Optional third-party providers are filtered automatically. If a package such as
`dion`, `gram-newton-schulz`, `liger-kernel`, `quack-kernels`,
`flash-linear-attention`, or `causal-conv1d` is not installed, the corresponding
provider is skipped.

## Running Benchmarks on Modal

Use the standalone Modal runner to execute benchmarks on a remote GPU and merge the
returned CSV rows into the local benchmark data file:

```bash
uv run --extra modal python scripts/run_modal_benchmarks.py \
  --gpu L40S \
  --benchmark "src/benchmarks/causal_conv1d_bench.py --mode full"
```

Pass `--benchmark` more than once to run a set of benchmark scripts in the same Modal
job:

```bash
uv run --extra modal python scripts/run_modal_benchmarks.py \
  --gpu H100 \
  --benchmark "src/benchmarks/chunked_linear_cross_entropy_bench.py --mode forward" \
  --benchmark "src/benchmarks/logprobs_bench.py --mode forward"
```

If no benchmark is specified, the runner executes all known benchmark entry points with
their default options. The Modal image installs the project dependencies plus the
`thirdparty` optional dependency group so optional providers are available remotely.
Use `--extra` to choose a narrower dependency set for the remote image:

```bash
uv run --extra modal python scripts/run_modal_benchmarks.py \
  --gpu H100 \
  --extra quack \
  --benchmark "src/benchmarks/chunked_linear_cross_entropy_bench.py --providers quack --mode full"
```

The `--extra` flag can be repeated, and the `modal` extra is added automatically.
Install `causal-conv1d` explicitly with `--extra causal-conv1d` when running that
comparison provider. Add `--plot-after` to regenerate documentation plots after the
remote CSV rows are merged.

## Metrics

Speed benchmarks record median runtime plus p20/p80 timing quantiles. Memory
benchmarks record peak allocated CUDA memory where the script supports memory
measurement.

The docs generally present benchmark plots in this order:

1. `full`
2. `forward`
3. `backward`

That keeps the end-to-end training measurement visible first while still showing
which pass contributes most to the result.

## Generating Plots

After benchmarks have written CSV rows, regenerate plots with:

```bash
uv run python src/benchmarks/plot_from_csv.py \
  --csv src/benchmarks/data/benchmark_data.csv
```

`plot_from_csv.py` emits PNG plots for one-dimensional sweeps, memory-speed Pareto
PNGs for kernels with paired speed and memory rows, and Plotly HTML plots for
two-dimensional sweeps. The documentation uses the interactive HTML plots when they are
available.

By default, generated plots are written directly into the docs assets:

```text
docs/public/assets/plots/
docs/public/assets/plots_html/
```

Then reference them from a kernel page with the shared iframe styling:

```html
<iframe
  class="plot-frame"
  title="Chunked Linear Cross-Entropy full speed"
  src="../../assets/plots_html/chunked_linear_cross_entropy_full_speed_batch_size=8192_dtype=bfloat16_NVIDIA_GeForce_RTX_4090.html"
  scrolling="no"
></iframe>
```

## Reproducibility Notes

Benchmark results depend on GPU model, installed third-party kernels, PyTorch version,
CUDA version, and compile/autotune state. The CSV records `gpu_name` and `git_commit`,
but comparisons are most meaningful when collected on the same machine with the same
dependency set.
