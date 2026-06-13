# Both greyhound and liger fuse gradient computation into the forward pass
# (the kernel overwrites the logits tensor in-place with dlogits), so only
# forward mode is benchmarked — there is no separate backward to measure.

import itertools
import json
import traceback

import click
import torch
from rich.table import Table

from benchmarks.third_party.liger import liger_cross_entropy
from benchmarks.third_party.quack import quack_cross_entropy
from benchmarks.third_party.torch import torch_cross_entropy
from benchmarks.utils import (
    DEVICE,
    DTYPE_MAP,
    QUANTILES,
    console,
    filter_providers,
    get_dtype,
    get_git_commit,
    get_gpu_name,
    print_benchmark_progress,
    print_speedup_table_2d,
    save_heatmap,
    save_results_csv,
)
from greyhound.kernels.cross_entropy import cross_entropy_with_grad_kernel
from greyhound.kernels.tuning_utils import do_bench


def _greyhound_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "sum",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    """Wrapper around cross_entropy_with_grad_kernel for benchmarking.

    The kernel computes cross-entropy loss and writes the gradient (dlogits)
    back into the ``logits`` tensor in-place.
    """
    ce_sum, z_sum, n_valid = cross_entropy_with_grad_kernel(
        logits, target, z_loss_multiplier, ignore_index
    )
    loss = ce_sum + z_loss_multiplier * z_sum
    if reduction == "mean":
        loss = loss / n_valid.clamp(min=1.0)
    return loss


IMPLEMENTATIONS = {
    "torch.cross_entropy": torch_cross_entropy,
    "liger": liger_cross_entropy,
    "quack": quack_cross_entropy,
    "greyhound": _greyhound_cross_entropy,
}
PROVIDERS = list(IMPLEMENTATIONS.keys())


def compute_gbps(batch_size: int, vocab_size: int, dtype: torch.dtype, median_ms: float) -> float:
    """Compute GB/s throughput for the cross-entropy + gradient operation.

    Both kernels read logits [N, V] + labels [N] and write dlogits [N, V].
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    logit_bytes = batch_size * vocab_size * elem_size
    label_bytes = batch_size * 8  # int64
    total_bytes = 2 * logit_bytes + label_bytes
    return total_bytes / (median_ms * 1e-3) / 1e9


def bench_speed_ce(
    provider: str,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    reduction: str,
    z_loss_multiplier: float,
) -> tuple[float, float, float]:
    # Liger only computes and stores dlogits in its forward kernel when logits require grad.
    # Greyhound always overwrites logits with dlogits, so require gradients here to keep the
    # raw-kernel benchmark comparing CE + gradient against CE + gradient.
    logits = torch.randn(batch_size, vocab_size, device=DEVICE, dtype=dtype, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)

    fn = IMPLEMENTATIONS[provider]

    def bench_fn() -> torch.Tensor:
        return fn(logits, target, reduction=reduction, z_loss_multiplier=z_loss_multiplier)

    bench_fn()  # warmup
    return do_bench(bench_fn, quantiles=QUANTILES)


def bench_mem_ce(
    provider: str,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    reduction: str,
    z_loss_multiplier: float,
) -> float:
    """Measure peak GPU memory (in MB) for the cross-entropy operation."""
    logits = torch.randn(batch_size, vocab_size, device=DEVICE, dtype=dtype, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)

    fn = IMPLEMENTATIONS[provider]

    def bench_fn() -> torch.Tensor:
        return fn(logits, target, reduction=reduction, z_loss_multiplier=z_loss_multiplier)

    bench_fn()  # warmup

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)
    bench_fn()
    torch.cuda.synchronize(DEVICE)
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    return peak_bytes / (1024 * 1024)


@click.command()
@click.option(
    "--providers",
    default=",".join(PROVIDERS),
    help="Comma-separated providers to benchmark",
)
@click.option(
    "--batch-sizes",
    default="2048,4096,8192",
    help="Comma-separated batch sizes / token counts to benchmark",
)
@click.option(
    "--vocab-sizes",
    default="65536,100352,128256,152064,200064",
    help="Comma-separated vocabulary sizes to benchmark",
)
@click.option(
    "--dtype",
    default="bfloat16",
    type=click.Choice(list(DTYPE_MAP.keys())),
    help="Data type for tensors",
)
@click.option(
    "--z-loss-multiplier",
    default=0.0,
    type=float,
    help="Z-loss multiplier (0.0 to disable, typical value 1e-3)",
)
@click.option(
    "--reduction",
    default="sum",
    type=click.Choice(["sum", "mean"]),
    show_default=True,
    help="Loss reduction to use for all providers",
)
@click.option(
    "--plot",
    default=None,
    type=click.Path(),
    help="Save a speed plot of results to PATH",
)
@click.option(
    "--mem-plot",
    default=None,
    type=click.Path(),
    help="Benchmark peak memory and save plot to PATH",
)
def main(
    providers: str,
    batch_sizes: str,
    vocab_sizes: str,
    dtype: str,
    z_loss_multiplier: float,
    reduction: str,
    plot: str | None,
    mem_plot: str | None,
) -> None:
    dt = get_dtype(dtype)
    batch_size_list = [int(s) for s in batch_sizes.split(",")]
    vocab_size_list = [int(s) for s in vocab_sizes.split(",")]

    provider_list = filter_providers([p.strip() for p in providers.split(",")])
    if not provider_list:
        console.print("[red]No providers available to benchmark[/red]")
        return

    console.print()
    console.rule("[bold blue]Cross-Entropy + Gradient Benchmark[/bold blue]")
    console.print(
        f"[dim]batch_sizes={batch_size_list}, vocab_sizes={vocab_size_list}, "
        f"dtype={dtype}, reduction={reduction}, z_loss_multiplier={z_loss_multiplier}[/dim]"
    )
    console.print()

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("batch_size", justify="right")
    table.add_column("vocab_size", justify="right")
    table.add_column("Median (ms)", justify="right", style="green")
    table.add_column("GB/s", justify="right", style="green")
    table.add_column("P20 (ms)", justify="right", style="dim")
    table.add_column("P80 (ms)", justify="right", style="dim")

    results = []
    for provider, batch_size, vocab_size in itertools.product(
        provider_list, batch_size_list, vocab_size_list
    ):
        try:
            times = bench_speed_ce(
                provider=provider,
                batch_size=batch_size,
                vocab_size=vocab_size,
                dtype=dt,
                reduction=reduction,
                z_loss_multiplier=z_loss_multiplier,
            )
            median, p20, p80 = times
        except Exception as e:
            console.print(
                f"[red]Error benchmarking {provider} with "
                f"batch_size={batch_size}, vocab_size={vocab_size}: {e}[/red]"
            )
            console.print(traceback.format_exc())
            median, p20, p80 = float("nan"), float("nan"), float("nan")

        gbps = compute_gbps(batch_size, vocab_size, dt, median)
        results.append(
            {
                "provider": provider,
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "median": median,
                "gbps": gbps,
                "p20": p20,
                "p80": p80,
            }
        )
        table.add_row(
            provider,
            str(batch_size),
            str(vocab_size),
            f"{median:.4f}",
            f"{gbps:.1f}",
            f"{p20:.4f}",
            f"{p80:.4f}",
        )
        print_benchmark_progress(
            provider=provider,
            batch_size=batch_size,
            vocab_size=vocab_size,
            median_ms=median,
            gbps=gbps,
            p20_ms=p20,
            p80_ms=p80,
        )

    console.print(table)
    print_speedup_table_2d(
        results, provider_list, "batch_size", batch_size_list, "vocab_size", vocab_size_list
    )

    gpu_name = get_gpu_name()
    git_commit = get_git_commit()
    extra = json.dumps(
        {
            "dtype": dtype,
            "reduction": reduction,
            "z_loss_multiplier": z_loss_multiplier,
        },
        sort_keys=True,
    )
    csv_rows = [
        {
            "kernel_name": "cross_entropy",
            "kernel_provider": r["provider"],
            "operation_mode": "forward",
            "metric_name": "speed",
            "metric_unit": "ms",
            "x_name": "batch_size,vocab_size",
            "x_value": f"{r['batch_size']},{r['vocab_size']}",
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
        save_heatmap(
            results=results,
            providers=provider_list,
            row_key="batch_size",
            row_values=batch_size_list,
            col_key="vocab_size",
            col_values=vocab_size_list,
            y_key="gbps",
            title=f"Cross-Entropy + Gradient ({dtype})",
            y_label="Memory Bandwidth (GB/s)",
            output_path=plot,
        )

    # --- Memory benchmark ---
    console.print()
    console.rule("[bold blue]Memory Benchmark[/bold blue]")
    for batch_size, vocab_size in itertools.product(batch_size_list, vocab_size_list):
        logit_bytes = batch_size * vocab_size * torch.tensor([], dtype=dt).element_size()
        logit_mb = logit_bytes / (1024 * 1024)
        console.print(
            f"[dim]  logit tensor (B={batch_size}, V={vocab_size}, {dtype}): "
            f"{logit_mb:.1f} MB[/dim]"
        )
    console.print()

    mem_table = Table(title="Peak Memory Usage", show_header=True, header_style="bold magenta")
    mem_table.add_column("Provider", style="cyan")
    mem_table.add_column("batch_size", justify="right")
    mem_table.add_column("vocab_size", justify="right")
    mem_table.add_column("Peak Memory (MB)", justify="right", style="green")

    mem_results = []
    for provider, batch_size, vocab_size in itertools.product(
        provider_list, batch_size_list, vocab_size_list
    ):
        try:
            peak_mb = bench_mem_ce(
                provider=provider,
                batch_size=batch_size,
                vocab_size=vocab_size,
                dtype=dt,
                reduction=reduction,
                z_loss_multiplier=z_loss_multiplier,
            )
        except Exception as e:
            console.print(
                f"[red]Error benchmarking memory for {provider} "
                f"with batch_size={batch_size}, vocab_size={vocab_size}: {e}[/red]"
            )
            peak_mb = float("nan")

        mem_results.append(
            {
                "provider": provider,
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "peak_mb": peak_mb,
            }
        )
        mem_table.add_row(
            provider,
            str(batch_size),
            str(vocab_size),
            f"{peak_mb:.1f}",
        )
        print_benchmark_progress(
            "memory",
            provider=provider,
            batch_size=batch_size,
            vocab_size=vocab_size,
            peak_mb=peak_mb,
        )

    console.print(mem_table)

    mem_csv_rows = [
        {
            "kernel_name": "cross_entropy",
            "kernel_provider": r["provider"],
            "operation_mode": "forward",
            "metric_name": "memory",
            "metric_unit": "MB",
            "x_name": "batch_size,vocab_size",
            "x_value": f"{r['batch_size']},{r['vocab_size']}",
            "y_value_50": r["peak_mb"],
            "y_value_20": "",
            "y_value_80": "",
            "extra_benchmark_config_str": extra,
            "gpu_name": gpu_name,
            "git_commit": git_commit,
        }
        for r in mem_results
    ]
    save_results_csv(mem_csv_rows)

    if mem_plot:
        save_heatmap(
            results=mem_results,
            providers=provider_list,
            row_key="batch_size",
            row_values=batch_size_list,
            col_key="vocab_size",
            col_values=vocab_size_list,
            y_key="peak_mb",
            title=f"Cross-Entropy + Gradient Peak Memory ({dtype})",
            y_label="Peak Memory (MB)",
            output_path=mem_plot,
        )


if __name__ == "__main__":
    main()
