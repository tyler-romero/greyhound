import itertools
import json
import traceback

import click
import torch
import torch._dynamo
from rich.table import Table

from benchmarks.third_party.dion import dion_newton_schulz, has_dion
from benchmarks.third_party.gram_newton_schulz import (
    gram_newton_schulz,
    has_gram_newton_schulz,
)
from benchmarks.third_party.torch import torch_newton_schulz
from benchmarks.utils import (
    DEVICE,
    DTYPE_MAP,
    QUANTILES,
    compile_for_benchmark,
    console,
    get_dtype,
    get_git_commit,
    get_gpu_name,
    print_benchmark_progress,
    print_speedup_table,
    save_plot,
    save_results_csv,
)
from greyhound.bonus.newton_schultz import has_quack_symmetric_gemm, orthogonalize_via_newton_schulz
from greyhound.kernels.tuning_utils import do_bench

torch._dynamo.config.cache_size_limit = 256

# We take Qwen3 weight shapes as representative of typical transformer matrice
QWEN3_MATRIX_SHAPES = (
    # Qwen3-0.6B: h=1024, ffn=3072
    "1024x1024",
    "3072x1024",
    "1024x3072",
    # Qwen3-4B: h=2560, kv=1024, ffn=9728
    "2560x2560",
    "1024x2560",
    "9728x2560",
    # Qwen3-8B: h=4096, kv=1024, ffn=12288
    "4096x4096",
    "1024x4096",
    "12288x4096",
    # Qwen3-14B/32B: h=5120, kv=1024, ffn=17408/25600
    "5120x5120",
    "1024x5120",
    "17408x5120",
    "25600x5120",
)
QWEN3_MATRIX_SHAPES_STR = ",".join(QWEN3_MATRIX_SHAPES)

IMPLEMENTATIONS = {
    "torch-eager": torch_newton_schulz,
    "torch-compile": torch_newton_schulz,
    "dion": dion_newton_schulz,
    "gram-newton-schulz": gram_newton_schulz,
    "greyhound + quack": orthogonalize_via_newton_schulz,
}
PROVIDERS = list(IMPLEMENTATIONS.keys())


def filter_newton_schulz_providers(providers: list[str]) -> list[str]:
    available = []
    for provider in providers:
        if provider == "dion" and not has_dion:
            console.print("[yellow]Warning: Dion not installed, skipping dion[/yellow]")
            continue
        if provider == "gram-newton-schulz" and not has_gram_newton_schulz:
            console.print(
                "[yellow]Warning: gram-newton-schulz not installed, "
                "skipping gram-newton-schulz[/yellow]"
            )
            continue
        if provider == "greyhound + quack" and not has_quack_symmetric_gemm(DEVICE):
            console.print(
                "[yellow]Warning: Quack symmetric GEMM is not available on this GPU, "
                "skipping greyhound + quack[/yellow]"
            )
            continue
        available.append(provider)
    return available


def compute_tflops(
    batch_size: int,
    m: int,
    n: int,
    iterations: int,
    median_ms: float,
) -> float:
    inner_rows = min(m, n)
    inner_cols = max(m, n)
    # Per iteration: X@X.T, A@A, and B@X. Count dense GEMM flops for provider-neutral
    # comparison even though Greyhound and Dion avoid about half the first two stores/work.
    flops_per_iter = (
        2 * inner_rows * inner_rows * inner_cols
        + 2 * inner_rows * inner_rows * inner_rows
        + 2 * inner_rows * inner_rows * inner_cols
    )
    total_flops = batch_size * iterations * flops_per_iter
    return total_flops / (median_ms * 1e-3) / 1e12


def bench_speed_newton_schulz(
    provider: str,
    batch_size: int,
    m: int,
    n: int,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    fn = IMPLEMENTATIONS.get(provider)
    if fn is None:
        raise ValueError(f"Invalid provider: {provider}")

    g = torch.randn(batch_size, m, n, device=DEVICE, dtype=dtype)
    if batch_size == 1:
        g = g.squeeze(0)

    if provider == "torch-compile":
        fn = compile_for_benchmark(fn)

    def bench_fn():
        return fn(g)

    bench_fn()
    return do_bench(bench_fn, quantiles=QUANTILES)


def parse_matrix_shapes(matrix_shapes: str) -> list[tuple[int, int]]:
    shapes = []
    for shape_str in matrix_shapes.split(","):
        shape_str = shape_str.strip().lower().replace(" ", "")
        if not shape_str:
            continue
        parts = shape_str.split("x")
        if len(parts) != 2:
            raise click.BadParameter(f"expected comma-separated MxN shapes, got {shape_str!r}")
        m, n = (int(part) for part in parts)
        if m <= 0 or n <= 0:
            raise click.BadParameter(f"matrix dimensions must be positive, got {shape_str!r}")
        shapes.append((m, n))
    if not shapes:
        raise click.BadParameter("expected at least one matrix shape")
    return shapes


@click.command()
@click.option(
    "--providers",
    default=",".join(PROVIDERS),
    help="Comma-separated providers to benchmark",
)
@click.option("--batch-sizes", default="1,2,4,8,16", help="Comma-separated batch sizes")
@click.option(
    "--matrix-shapes",
    default=QWEN3_MATRIX_SHAPES_STR,
    help=(
        "Comma-separated matrix shapes as MxN. Defaults are representative Qwen3 "
        "attention, GQA KV, and MLP parameter shapes."
    ),
)
@click.option(
    "--dtype",
    default="bfloat16",
    type=click.Choice(list(DTYPE_MAP.keys())),
    help="Data type for input tensors",
)
@click.option(
    "--plot",
    default=None,
    type=click.Path(),
    help="Save a plot of results to PATH",
)
def main(
    providers: str,
    batch_sizes: str,
    matrix_shapes: str,
    dtype: str,
    plot: str | None,
) -> None:
    dt = get_dtype(dtype)
    batch_size_list = [int(s) for s in batch_sizes.split(",")]
    shape_list = parse_matrix_shapes(matrix_shapes)
    shape_labels = [f"{m}x{n}" for m, n in shape_list]

    provider_list = filter_newton_schulz_providers([p.strip() for p in providers.split(",")])
    if not provider_list:
        console.print("[red]No providers available to benchmark[/red]")
        return

    console.print()
    console.rule("[bold blue]Newton-Schulz Benchmark[/bold blue]")
    console.print(
        f"[dim]batch_sizes={batch_size_list}, matrix_shapes={shape_labels}, "
        f"dtype={dtype}, device={DEVICE}, providers={provider_list}[/dim]"
    )
    console.print()

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("batch", justify="right")
    table.add_column("m", justify="right")
    table.add_column("n", justify="right")
    table.add_column("Median (ms)", justify="right", style="green")
    table.add_column("TFLOP/s", justify="right", style="green")
    table.add_column("P20 (ms)", justify="right", style="dim")
    table.add_column("P80 (ms)", justify="right", style="dim")

    results = []
    for batch_size, (m, n), provider in itertools.product(
        batch_size_list, shape_list, provider_list
    ):
        try:
            median, p20, p80 = bench_speed_newton_schulz(
                provider=provider,
                batch_size=batch_size,
                m=m,
                n=n,
                dtype=dt,
            )
        except Exception as e:
            console.print(
                f"[red]Error benchmarking {provider} with batch={batch_size}, "
                f"m={m}, n={n}: {e}[/red]"
            )
            console.print(traceback.format_exc())
            median, p20, p80 = float("nan"), float("nan"), float("nan")

        shape_label = f"{m}x{n}"
        tflops = compute_tflops(batch_size, m, n, 5, median)
        result = {
            "provider": provider,
            "batch_size": batch_size,
            "m": m,
            "n": n,
            "matrix_shape": shape_label,
            "shape": f"b{batch_size}:{shape_label}",
            "median": median,
            "tflops": tflops,
            "p20": p20,
            "p80": p80,
        }
        results.append(result)
        table.add_row(
            provider,
            str(batch_size),
            str(m),
            str(n),
            f"{median:.4f}",
            f"{tflops:.2f}",
            f"{p20:.4f}",
            f"{p80:.4f}",
        )
        print_benchmark_progress(
            provider=provider,
            batch_size=batch_size,
            m=m,
            n=n,
            median_ms=median,
            tflops=tflops,
            p20_ms=p20,
            p80_ms=p80,
        )

    console.print(table)
    shapes = [f"b{b}:{shape}" for b, shape in itertools.product(batch_size_list, shape_labels)]
    print_speedup_table(results, provider_list, "shape", shapes)  # ty:ignore[invalid-argument-type]

    gpu_name = get_gpu_name()
    git_commit = get_git_commit()
    extra = json.dumps({"dtype": dtype}, sort_keys=True)
    csv_rows = [
        {
            "kernel_name": "newton_schulz",
            "kernel_provider": r["provider"],
            "operation_mode": "forward",
            "metric_name": "speed",
            "metric_unit": "ms",
            "x_name": "batch_size,m,n",
            "x_value": f"{r['batch_size']},{r['m']},{r['n']}",
            "y_value_50": r["median"],
            "y_value_20": r["p20"],
            "y_value_80": r["p80"],
            "extra_benchmark_config_str": extra,
            "gpu_name": gpu_name,
            "git_commit": git_commit,
        }
        for r in results
    ]
    save_results_csv(csv_rows)

    if plot:
        if len(batch_size_list) != 1:
            console.print(
                "[yellow]Plotting Newton-Schulz requires exactly one batch size; "
                "skipping plot[/yellow]"
            )
            return
        save_plot(
            results=results,
            providers=provider_list,
            size_key="matrix_shape",
            sizes=shape_labels,  # ty:ignore[invalid-argument-type]
            y_key="tflops",
            title=f"Newton-Schulz ({dtype})",
            y_label="TFLOP/s",
            x_label="matrix shape (m x n)",
            output_path=plot,
        )


if __name__ == "__main__":
    main()
