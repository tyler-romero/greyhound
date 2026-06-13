import itertools
import json
import traceback
from typing import Any, Callable, cast

import click
import torch
from rich.table import Table

from benchmarks.third_party.causal_conv1d import causal_conv1d_fn
from benchmarks.third_party.fla import fla_causal_conv1d
from benchmarks.third_party.torch import torch_causal_conv1d
from benchmarks.utils import (
    DEVICE,
    DTYPE_MAP,
    QUANTILES,
    compile_for_benchmark,
    console,
    filter_providers,
    get_git_commit,
    get_gpu_name,
    print_benchmark_progress,
    print_speedup_table_2d,
    save_heatmap,
    save_results_csv,
)
from greyhound.kernels.tuning_utils import do_bench
from greyhound.nn.functional import causal_conv1d as greyhound_causal_conv1d

# Disable recompile limit since we benchmark with varying tensor sizes
torch._dynamo.config.cache_size_limit = 256


IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "torch-eager": torch_causal_conv1d,
    "torch-compile": torch_causal_conv1d,
    "causal-conv1d": causal_conv1d_fn,
    "fla": fla_causal_conv1d,
    "greyhound": greyhound_causal_conv1d,
}
PROVIDERS = list(IMPLEMENTATIONS.keys())


def bench_speed_causal_conv1d(
    provider: str,
    operation_mode: str,
    batch_size: int,
    dim: int,
    seqlen: int,
    width: int,
    activation: str | None,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    requires_grad = operation_mode != "forward"

    x = torch.randn(
        batch_size, dim, seqlen, device=DEVICE, dtype=dtype, requires_grad=requires_grad
    )
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype, requires_grad=requires_grad)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype, requires_grad=requires_grad)

    fn = IMPLEMENTATIONS.get(provider)
    if fn is None:
        raise ValueError(f"Invalid provider: {provider}")

    if provider != "torch-eager":
        fn = compile_for_benchmark(fn)

    def fwd():
        return fn(x, weight, bias=bias, activation=cast(Any, activation))

    # causal_conv1d preserves [B, D, T], so allocate the upstream gradient before
    # timing without running an extra forward pass or timing random-number generation.
    grad_out = torch.randn_like(x) if requires_grad else None

    def full():
        y = fwd()
        assert grad_out is not None
        y.backward(grad_out, retain_graph=True)

    if operation_mode == "forward":
        bench_fn = fwd
    elif operation_mode == "backward":
        y = fwd()
        assert grad_out is not None

        def bench_fn():
            return y.backward(grad_out, retain_graph=True)
    else:  # full
        bench_fn = full

    bench_fn()  # compile + warmup
    return do_bench(bench_fn, grad_to_none=[x, weight, bias], quantiles=QUANTILES)


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
@click.option("--batch-size", default=8, help="Batch size")
@click.option(
    "--dims",
    default="1024,2048,4096,8192",
    help="Comma-separated channel dimensions to benchmark",
)
@click.option("--width", default=4, help="Convolution kernel width")
@click.option(
    "--seqlens",
    default="2048,4096,8192",
    help="Comma-separated sequence lengths to benchmark",
)
@click.option(
    "--activation",
    default="silu",
    type=click.Choice(["silu", "none"]),
    help="Activation function",
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
def main(
    providers: str,
    mode: str,
    batch_size: int,
    dims: str,
    width: int,
    seqlens: str,
    activation: str,
    dtype: str,
    plot: str | None,
) -> None:
    dt = DTYPE_MAP[dtype]
    dim_list = [int(s) for s in dims.split(",")]
    seqlen_list = [int(s) for s in seqlens.split(",")]
    act: str | None = None if activation == "none" else activation

    provider_list = filter_providers([p.strip() for p in providers.split(",")])
    if not provider_list:
        console.print("[red]No providers available to benchmark[/red]")
        return

    console.print()
    console.rule(f"[bold blue]Causal Conv1d Benchmark ({mode} pass)[/bold blue]")
    console.print(
        f"[dim]batch_size={batch_size}, dims={dim_list}, seqlens={seqlen_list}, width={width}, "
        f"activation={activation}, dtype={dtype}[/dim]"
    )
    console.print()

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("dim", justify="right")
    table.add_column("seqlen", justify="right")
    table.add_column("Median (ms)", justify="right", style="green")
    table.add_column("P20 (ms)", justify="right", style="dim")
    table.add_column("P80 (ms)", justify="right", style="dim")

    results = []
    for provider, dim, seqlen in itertools.product(provider_list, dim_list, seqlen_list):
        try:
            median, p20, p80 = bench_speed_causal_conv1d(
                provider=provider,
                operation_mode=mode,
                batch_size=batch_size,
                dim=dim,
                seqlen=seqlen,
                width=width,
                activation=act,
                dtype=dt,
            )
        except Exception as e:
            console.print(
                f"[red]Error benchmarking {provider} with dim={dim}, seqlen={seqlen}: {e}[/red]"
            )
            console.print(traceback.format_exc())
            median, p20, p80 = float("nan"), float("nan"), float("nan")

        results.append(
            {
                "provider": provider,
                "dim": dim,
                "seqlen": seqlen,
                "median": median,
                "p20": p20,
                "p80": p80,
            }
        )
        table.add_row(
            provider,
            str(dim),
            str(seqlen),
            f"{median:.4f}",
            f"{p20:.4f}",
            f"{p80:.4f}",
        )
        print_benchmark_progress(
            provider=provider,
            dim=dim,
            seqlen=seqlen,
            median_ms=median,
            p20_ms=p20,
            p80_ms=p80,
        )

    console.print(table)
    print_speedup_table_2d(results, provider_list, "dim", dim_list, "seqlen", seqlen_list)

    gpu_name = get_gpu_name()
    git_commit = get_git_commit()
    extra = json.dumps(
        {
            "activation": activation,
            "batch_size": batch_size,
            "dtype": dtype,
            "width": width,
        },
        sort_keys=True,
    )
    csv_rows = [
        {
            "kernel_name": "causal_conv1d",
            "kernel_provider": r["provider"],
            "operation_mode": mode,
            "metric_name": "speed",
            "metric_unit": "ms",
            "x_name": "dim,seqlen",
            "x_value": f"{r['dim']},{r['seqlen']}",
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
            row_key="dim",
            row_values=dim_list,
            col_key="seqlen",
            col_values=seqlen_list,
            y_key="median",
            title=f"Causal Conv1d ({mode}, {dtype})",
            y_label="Time (ms)",
            output_path=plot,
        )


if __name__ == "__main__":
    main()
