import itertools
import json
import traceback

import click
import torch
from rich.table import Table

from benchmarks.third_party.torch import torch_selective_log_softmax
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
    print_speedup_table_2d,
    save_heatmap,
    save_results_csv,
)
from greyhound.kernels.tuning_utils import do_bench
from greyhound.nn.functional import selective_log_softmax

torch._dynamo.config.cache_size_limit = 256

IMPLEMENTATIONS = {
    "torch-eager": torch_selective_log_softmax,
    "torch-compile": torch_selective_log_softmax,
    "greyhound": selective_log_softmax,
}
PROVIDERS = list(IMPLEMENTATIONS.keys())


def compute_gbps(
    batch_size: int, seq_len: int, vocab_size: int, dtype: torch.dtype, median_ms: float
) -> float:
    """Compute approximate GB/s for selected log-softmax.

    Greyhound reads logits once plus indices and writes selected log-probs. PyTorch eager
    materializes full log-probs, so this is a useful-op bandwidth metric rather than exact
    provider-specific traffic.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    rows = batch_size * seq_len
    logit_bytes = rows * vocab_size * elem_size
    index_bytes = rows * 8
    out_bytes = rows * elem_size
    return (logit_bytes + index_bytes + out_bytes) / (median_ms * 1e-3) / 1e9


def bench_speed_logprobs(
    provider: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    logits = torch.randn(batch_size, seq_len, vocab_size, device=DEVICE, dtype=dtype)
    index = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)

    fn = IMPLEMENTATIONS.get(provider)
    if fn is None:
        raise ValueError(f"Invalid provider: {provider}")

    if provider != "torch-eager":
        fn = compile_for_benchmark(fn)

    def bench_fn() -> torch.Tensor:
        return fn(logits, index)

    bench_fn()
    return do_bench(bench_fn, quantiles=QUANTILES)


def bench_mem_logprobs(
    provider: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> float:
    logits = torch.randn(batch_size, seq_len, vocab_size, device=DEVICE, dtype=dtype)
    index = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)

    fn = IMPLEMENTATIONS.get(provider)
    if fn is None:
        raise ValueError(f"Invalid provider: {provider}")

    if provider != "torch-eager":
        fn = compile_for_benchmark(fn)

    def bench_fn() -> torch.Tensor:
        return fn(logits, index)

    bench_fn()

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
@click.option("--batch-size", default=8, help="Batch size")
@click.option(
    "--seq-lens",
    default="2048,4096,8192",
    help="Comma-separated sequence lengths to benchmark",
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
    help="Data type for logits",
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
    batch_size: int,
    seq_lens: str,
    vocab_sizes: str,
    dtype: str,
    plot: str | None,
    mem_plot: str | None,
) -> None:
    dt = get_dtype(dtype)
    seq_len_list = [int(s) for s in seq_lens.split(",")]
    vocab_size_list = [int(s) for s in vocab_sizes.split(",")]
    provider_list = [p.strip() for p in providers.split(",")]
    if not provider_list:
        console.print("[red]No providers available to benchmark[/red]")
        return

    console.print()
    console.rule("[bold blue]Selective Log-Softmax Benchmark (forward pass)[/bold blue]")
    console.print(
        f"[dim]batch_size={batch_size}, seq_lens={seq_len_list}, "
        f"vocab_sizes={vocab_size_list}, dtype={dtype}[/dim]"
    )
    console.print()

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("seq_len", justify="right")
    table.add_column("vocab_size", justify="right")
    table.add_column("Median (ms)", justify="right", style="green")
    table.add_column("GB/s", justify="right", style="green")
    table.add_column("P20 (ms)", justify="right", style="dim")
    table.add_column("P80 (ms)", justify="right", style="dim")

    results = []
    for provider, seq_len, vocab_size in itertools.product(
        provider_list, seq_len_list, vocab_size_list
    ):
        try:
            median, p20, p80 = bench_speed_logprobs(
                provider=provider,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                dtype=dt,
            )
        except Exception as e:
            console.print(
                f"[red]Error benchmarking {provider} with "
                f"seq_len={seq_len}, vocab_size={vocab_size}: {e}[/red]"
            )
            console.print(traceback.format_exc())
            median, p20, p80 = float("nan"), float("nan"), float("nan")

        gbps = compute_gbps(batch_size, seq_len, vocab_size, dt, median)
        results.append(
            {
                "provider": provider,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "median": median,
                "gbps": gbps,
                "p20": p20,
                "p80": p80,
            }
        )
        table.add_row(
            provider,
            str(seq_len),
            str(vocab_size),
            f"{median:.4f}",
            f"{gbps:.1f}",
            f"{p20:.4f}",
            f"{p80:.4f}",
        )
        print_benchmark_progress(
            provider=provider,
            seq_len=seq_len,
            vocab_size=vocab_size,
            median_ms=median,
            gbps=gbps,
            p20_ms=p20,
            p80_ms=p80,
        )

    console.print(table)
    print_speedup_table_2d(
        results, provider_list, "seq_len", seq_len_list, "vocab_size", vocab_size_list
    )

    gpu_name = get_gpu_name()
    git_commit = get_git_commit()
    extra = json.dumps({"batch_size": batch_size, "dtype": dtype}, sort_keys=True)
    csv_rows = [
        {
            "kernel_name": "logprobs",
            "kernel_provider": r["provider"],
            "operation_mode": "forward",
            "metric_name": "speed",
            "metric_unit": "ms",
            "x_name": "seq_len,vocab_size",
            "x_value": f"{r['seq_len']},{r['vocab_size']}",
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
            row_key="seq_len",
            row_values=seq_len_list,
            col_key="vocab_size",
            col_values=vocab_size_list,
            y_key="gbps",
            title=f"Selective Log-Softmax ({dtype})",
            y_label="Useful Memory Bandwidth (GB/s)",
            output_path=plot,
        )

    console.print()
    console.rule("[bold blue]Memory Benchmark[/bold blue]")
    for seq_len, vocab_size in itertools.product(seq_len_list, vocab_size_list):
        logit_bytes = batch_size * seq_len * vocab_size * torch.tensor([], dtype=dt).element_size()
        logit_mb = logit_bytes / (1024 * 1024)
        console.print(
            f"[dim]  logits tensor (B={batch_size}, S={seq_len}, V={vocab_size}, {dtype}): "
            f"{logit_mb:.1f} MB[/dim]"
        )
    console.print()

    mem_table = Table(title="Peak Memory Usage", show_header=True, header_style="bold magenta")
    mem_table.add_column("Provider", style="cyan")
    mem_table.add_column("seq_len", justify="right")
    mem_table.add_column("vocab_size", justify="right")
    mem_table.add_column("Peak Memory (MB)", justify="right", style="green")

    mem_results = []
    for provider, seq_len, vocab_size in itertools.product(
        provider_list, seq_len_list, vocab_size_list
    ):
        try:
            peak_mb = bench_mem_logprobs(
                provider=provider,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                dtype=dt,
            )
        except Exception as e:
            console.print(
                f"[red]Error benchmarking memory for {provider} "
                f"with seq_len={seq_len}, vocab_size={vocab_size}: {e}[/red]"
            )
            peak_mb = float("nan")

        mem_results.append(
            {
                "provider": provider,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "peak_mb": peak_mb,
            }
        )
        mem_table.add_row(provider, str(seq_len), str(vocab_size), f"{peak_mb:.1f}")
        print_benchmark_progress(
            "memory",
            provider=provider,
            seq_len=seq_len,
            vocab_size=vocab_size,
            peak_mb=peak_mb,
        )

    console.print(mem_table)

    mem_csv_rows = [
        {
            "kernel_name": "logprobs",
            "kernel_provider": r["provider"],
            "operation_mode": "forward",
            "metric_name": "memory",
            "metric_unit": "MB",
            "x_name": "seq_len,vocab_size",
            "x_value": f"{r['seq_len']},{r['vocab_size']}",
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
            row_key="seq_len",
            row_values=seq_len_list,
            col_key="vocab_size",
            col_values=vocab_size_list,
            y_key="peak_mb",
            title=f"Selective Log-Softmax Peak Memory ({dtype})",
            y_label="Peak Memory (MB)",
            output_path=mem_plot,
        )


if __name__ == "__main__":
    main()
