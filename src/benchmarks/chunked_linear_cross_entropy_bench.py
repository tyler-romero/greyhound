import itertools
import json

import click
import torch
from rich.table import Table

from benchmarks.third_party.liger import liger_linear_cross_entropy
from benchmarks.third_party.quack import quack_linear_cross_entropy
from benchmarks.third_party.torch import torch_linear_cross_entropy
from benchmarks.utils import (
    DEVICE,
    DTYPE_MAP,
    QUANTILES,
    compile_for_benchmark,
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
from greyhound.kernels.tuning_utils import do_bench
from greyhound.nn.functional import chunked_linear_cross_entropy

# Disable recompile limit since we benchmark with varying tensor sizes
torch._dynamo.config.cache_size_limit = 256

IMPLEMENTATIONS = {
    "torch-eager": torch_linear_cross_entropy,
    "torch-compile": torch_linear_cross_entropy,
    "liger": liger_linear_cross_entropy,
    "quack": quack_linear_cross_entropy,
    "greyhound": chunked_linear_cross_entropy,
}
PROVIDERS = list(IMPLEMENTATIONS.keys())


def filter_chunked_linear_cross_entropy_providers(providers: list[str], mode: str) -> list[str]:
    available = filter_providers(providers)
    if "quack" in available and mode != "full":
        console.print(
            "[yellow]Warning: Quack chunked_linear_cross_entropy computes gradients in "
            "forward, skipping quack provider outside full mode[/yellow]"
        )
        available = [p for p in available if p != "quack"]
    if "quack" in available:
        capability = torch.cuda.get_device_capability(DEVICE)
        if capability[0] < 9:
            console.print(
                "[yellow]Warning: Quack chunked_linear_cross_entropy GEMM path requires "
                "SM90+, skipping quack provider on this GPU[/yellow]"
            )
            available = [p for p in available if p != "quack"]
    return available


def compute_tflops(
    batch_size: int, d_model: int, vocab_size: int, median_ms: float, mode: str
) -> float:
    """Compute TFLOPS for the chunked linear cross-entropy operation.

    The dominant cost is the linear projection: 2 * B * D * V FLOPs (forward).
    Backward is ~4 * B * D * V (recompute logits + grad_input + grad_weight matmuls).
    """
    if mode == "forward":
        flops = 2.0 * batch_size * d_model * vocab_size
    elif mode == "backward":
        flops = 4.0 * batch_size * d_model * vocab_size
    else:  # full
        flops = 6.0 * batch_size * d_model * vocab_size
    return flops / (median_ms * 1e-3) / 1e12


def bench_speed_chunked_linear_cross_entropy(
    provider: str,
    operation_mode: str,
    batch_size: int,
    d_model: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    # make benchmark fair for kernels that compute gradients in the forward pass
    requires_grad = operation_mode != "forward"
    if operation_mode == "backward" and provider in ("liger", "quack"):
        raise ValueError(f"{provider} backward pass cannot be benchmarked in isolation")
    if operation_mode == "forward" and provider == "quack":
        raise ValueError(
            "Quack chunked_linear_cross_entropy computes gradients in forward; "
            "benchmark it in full mode"
        )

    inputs = torch.randn(
        batch_size, d_model, device=DEVICE, dtype=dtype, requires_grad=requires_grad
    )
    weight = torch.randn(
        vocab_size, d_model, device=DEVICE, dtype=dtype, requires_grad=requires_grad
    )
    target = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)

    fn = IMPLEMENTATIONS.get(provider)
    if fn is None:
        raise ValueError(f"Invalid provider: {provider}")

    if provider not in ["torch-eager", "liger", "quack"]:
        fn = compile_for_benchmark(fn)

    def fwd():
        return fn(inputs, weight, target)

    def full():
        y = fwd()
        y.backward(retain_graph=True)

    if operation_mode == "forward":
        bench_fn = fwd
    elif operation_mode == "backward":
        y = fwd()

        def bench_fn():
            return y.backward(retain_graph=True)
    else:  # full
        bench_fn = full

    bench_fn()  # compile + warmup
    return do_bench(bench_fn, grad_to_none=[inputs, weight], quantiles=QUANTILES)


def bench_mem_chunked_linear_cross_entropy(
    provider: str,
    operation_mode: str,
    batch_size: int,
    d_model: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> float:
    """Measure peak GPU memory (in MB) for the chunked linear cross-entropy operation."""
    requires_grad = operation_mode != "forward"
    if operation_mode == "backward" and provider in ("liger", "quack"):
        raise ValueError(f"{provider} backward pass cannot be benchmarked in isolation")
    if operation_mode == "forward" and provider == "quack":
        raise ValueError(
            "Quack chunked_linear_cross_entropy computes gradients in forward; "
            "benchmark it in full mode"
        )

    inputs = torch.randn(
        batch_size, d_model, device=DEVICE, dtype=dtype, requires_grad=requires_grad
    )
    weight = torch.randn(
        vocab_size, d_model, device=DEVICE, dtype=dtype, requires_grad=requires_grad
    )
    target = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)

    fn = IMPLEMENTATIONS.get(provider)
    if fn is None:
        raise ValueError(f"Invalid provider: {provider}")

    if provider not in ["torch-eager", "liger", "quack"]:
        fn = compile_for_benchmark(fn)

    def fwd():
        return fn(inputs, weight, target)

    def full():
        y = fwd()
        y.backward(retain_graph=True)

    if operation_mode == "forward":
        bench_fn = fwd
    elif operation_mode == "backward":
        y = fwd()

        def bench_fn():
            return y.backward(retain_graph=True)
    else:  # full
        bench_fn = full

    # Warmup / compile
    bench_fn()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)
    bench_fn()
    torch.cuda.synchronize(DEVICE)
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    return peak_bytes / (1024 * 1024)  # Convert to MB


@click.command()
@click.option(
    "--providers",
    default=",".join(PROVIDERS),
    help="Comma-separated providers to benchmark",
)
@click.option(
    "--mode",
    default="full",
    type=click.Choice(["forward", "backward", "full"]),
    help="Operation mode",
)
@click.option("--batch-size", default=8192, help="Batch size (number of tokens)")
@click.option(
    "--d-models",
    default="1024,2048,4096,8192",
    help="Comma-separated hidden dimensions to benchmark",
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
    "--plot",
    default=None,
    type=click.Path(),
    help="Save a plot of results to PATH",
)
@click.option(
    "--mem-plot",
    default=None,
    type=click.Path(),
    help="Benchmark peak memory and save plot to PATH",
)
def main(
    providers: str,
    mode: str,
    batch_size: int,
    d_models: str,
    vocab_sizes: str,
    dtype: str,
    plot: str | None,
    mem_plot: str | None,
) -> None:
    dt = get_dtype(dtype)
    d_model_list = [int(s) for s in d_models.split(",")]
    vocab_size_list = [int(s) for s in vocab_sizes.split(",")]

    provider_list = filter_chunked_linear_cross_entropy_providers(
        [p.strip() for p in providers.split(",")], mode
    )
    if not provider_list:
        console.print("[red]No providers available to benchmark[/red]")
        return

    console.print()
    console.rule(f"[bold blue]Chunked Linear Cross-Entropy Benchmark ({mode} pass)[/bold blue]")
    console.print(
        f"[dim]batch_size={batch_size}, d_models={d_model_list}, "
        f"vocab_sizes={vocab_size_list}, dtype={dtype}[/dim]"
    )
    console.print()

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("d_model", justify="right")
    table.add_column("vocab_size", justify="right")
    table.add_column("Median (ms)", justify="right", style="green")
    table.add_column("TFLOPS", justify="right", style="green")
    table.add_column("P20 (ms)", justify="right", style="dim")
    table.add_column("P80 (ms)", justify="right", style="dim")

    results = []
    for provider, d_model, vocab_size in itertools.product(
        provider_list, d_model_list, vocab_size_list
    ):
        try:
            times = bench_speed_chunked_linear_cross_entropy(
                provider=provider,
                operation_mode=mode,
                batch_size=batch_size,
                d_model=d_model,
                vocab_size=vocab_size,
                dtype=dt,
            )
            median, p20, p80 = times
        except Exception as e:
            console.print(
                f"[red]Error benchmarking {provider} with "
                f"d_model={d_model}, vocab_size={vocab_size}: {e}[/red]"
            )
            median, p20, p80 = float("nan"), float("nan"), float("nan")
        tflops = compute_tflops(batch_size, d_model, vocab_size, median, mode)
        results.append(
            {
                "provider": provider,
                "d_model": d_model,
                "vocab_size": vocab_size,
                "median": median,
                "tflops": tflops,
                "p20": p20,
                "p80": p80,
            }
        )
        table.add_row(
            provider,
            str(d_model),
            str(vocab_size),
            f"{median:.4f}",
            f"{tflops:.1f}",
            f"{p20:.4f}",
            f"{p80:.4f}",
        )
        print_benchmark_progress(
            provider=provider,
            d_model=d_model,
            vocab_size=vocab_size,
            median_ms=median,
            tflops=tflops,
            p20_ms=p20,
            p80_ms=p80,
        )

    console.print(table)
    print_speedup_table_2d(
        results, provider_list, "d_model", d_model_list, "vocab_size", vocab_size_list
    )

    gpu_name = get_gpu_name()
    git_commit = get_git_commit()
    extra = json.dumps({"batch_size": batch_size, "dtype": dtype}, sort_keys=True)
    csv_rows = [
        {
            "kernel_name": "chunked_linear_cross_entropy",
            "kernel_provider": r["provider"],
            "operation_mode": mode,
            "metric_name": "speed",
            "metric_unit": "ms",
            "x_name": "d_model,vocab_size",
            "x_value": f"{r['d_model']},{r['vocab_size']}",
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
            row_key="d_model",
            row_values=d_model_list,
            col_key="vocab_size",
            col_values=vocab_size_list,
            y_key="tflops",
            title=f"Chunked Linear Cross-Entropy ({mode}, {dtype})",
            y_label="TFLOPS",
            output_path=plot,
        )

    console.print()
    console.rule("[bold blue]Memory Benchmark[/bold blue]")
    for d_model, vocab_size in itertools.product(d_model_list, vocab_size_list):
        logit_bytes = batch_size * vocab_size * 4  # float32
        logit_mb = logit_bytes / (1024 * 1024)
        console.print(
            f"[dim]  logit tensor (B={batch_size}, D={d_model}, V={vocab_size}, float32): "
            f"{logit_mb:.1f} MB[/dim]"
        )
    console.print()

    mem_table = Table(title="Peak Memory Usage", show_header=True, header_style="bold magenta")
    mem_table.add_column("Provider", style="cyan")
    mem_table.add_column("d_model", justify="right")
    mem_table.add_column("vocab_size", justify="right")
    mem_table.add_column("Peak Memory (MB)", justify="right", style="green")

    mem_results = []
    for provider, d_model, vocab_size in itertools.product(
        provider_list, d_model_list, vocab_size_list
    ):
        try:
            peak_mb = bench_mem_chunked_linear_cross_entropy(
                provider=provider,
                operation_mode=mode,
                batch_size=batch_size,
                d_model=d_model,
                vocab_size=vocab_size,
                dtype=dt,
            )
        except Exception as e:
            console.print(
                f"[red]Error benchmarking memory for {provider} "
                f"with d_model={d_model}, vocab_size={vocab_size}: {e}[/red]"
            )
            peak_mb = float("nan")

        mem_results.append(
            {
                "provider": provider,
                "d_model": d_model,
                "vocab_size": vocab_size,
                "peak_mb": peak_mb,
            }
        )
        mem_table.add_row(
            provider,
            str(d_model),
            str(vocab_size),
            f"{peak_mb:.1f}",
        )
        print_benchmark_progress(
            "memory",
            provider=provider,
            d_model=d_model,
            vocab_size=vocab_size,
            peak_mb=peak_mb,
        )

    console.print(mem_table)

    mem_csv_rows = [
        {
            "kernel_name": "chunked_linear_cross_entropy",
            "kernel_provider": r["provider"],
            "operation_mode": mode,
            "metric_name": "memory",
            "metric_unit": "MB",
            "x_name": "d_model,vocab_size",
            "x_value": f"{r['d_model']},{r['vocab_size']}",
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
            row_key="d_model",
            row_values=d_model_list,
            col_key="vocab_size",
            col_values=vocab_size_list,
            y_key="peak_mb",
            title=f"Chunked Linear Cross-Entropy Peak Memory ({mode}, {dtype})",
            y_label="Peak Memory (MB)",
            output_path=mem_plot,
        )


if __name__ == "__main__":
    main()
