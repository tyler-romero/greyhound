import csv
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from benchmarks.third_party.causal_conv1d import has_causal_conv1d
from benchmarks.third_party.fla import has_fla
from benchmarks.third_party.liger import has_liger
from benchmarks.third_party.quack import has_quack
from greyhound.utils import get_default_device

console = Console()

DEVICE = get_default_device()

DEFAULT_BENCHMARK_CSV = Path(__file__).parent / "data" / "benchmark_data.csv"
BENCHMARK_CSV = Path(os.environ.get("GREYHOUND_BENCHMARK_CSV", DEFAULT_BENCHMARK_CSV))

CSV_COLUMNS = [
    "kernel_name",
    "kernel_provider",
    "operation_mode",
    "metric_name",
    "metric_unit",
    "x_name",
    "x_value",
    "y_value_50",
    "y_value_20",
    "y_value_80",
    "extra_benchmark_config_str",
    "gpu_name",
    "git_commit",
]

CSV_KEY_COLUMNS = [
    "kernel_name",
    "kernel_provider",
    "operation_mode",
    "metric_name",
    "x_name",
    "x_value",
    "extra_benchmark_config_str",
    "gpu_name",
]

QUANTILES = [0.5, 0.2, 0.8]

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def compile_for_benchmark(obj: Any) -> Any:
    """Compile a benchmark callable/module using the configured compile mode."""
    mode = os.environ.get("GREYHOUND_TORCH_COMPILE_MODE", "default")
    if mode in {"", "default"}:
        return torch.compile(obj)
    if mode in {"none", "disabled"}:
        return obj
    return torch.compile(obj, mode=mode)


def get_dtype(dtype_str: str) -> torch.dtype:
    return DTYPE_MAP[dtype_str]


def filter_providers(providers: list[str]) -> list[str]:
    """Filter out providers that are not installed, printing warnings."""
    available = list(providers)
    if "liger" in available and not has_liger:
        console.print(
            "[yellow]Warning: Liger Kernel not installed, skipping liger provider[/yellow]"
        )
        available = [p for p in available if p != "liger"]
    if "quack" in available and not has_quack:
        console.print("[yellow]Warning: Quack not installed, skipping quack provider[/yellow]")
        available = [p for p in available if p != "quack"]
    if "fla" in available and not has_fla:
        console.print("[yellow]Warning: FLA not installed, skipping fla provider[/yellow]")
        available = [p for p in available if p != "fla"]
    if "causal-conv1d" in available and not has_causal_conv1d:
        console.print(
            "[yellow]Warning: causal-conv1d not installed, skipping causal-conv1d provider[/yellow]"
        )
        available = [p for p in available if p != "causal-conv1d"]
    return available


def print_speedup_table(
    results: list[dict[str, Any]],
    providers: list[str],
    size_key: str,
    sizes: list[int],
) -> None:
    """Print a speedup summary table comparing providers against the torch-compile baseline."""
    if len(providers) <= 1 or not results:
        return

    console.print()
    speedup_table = Table(
        title="Speedup Summary (vs torch-compile baseline)",
        show_header=True,
        header_style="bold magenta",
    )
    speedup_table.add_column(size_key, justify="right")
    speedup_table.add_column("Provider", style="cyan")
    speedup_table.add_column("Speedup", justify="right", style="green")

    for size in sizes:
        baseline = next(
            (r for r in results if r["provider"] == "torch-compile" and r[size_key] == size),
            None,
        )
        if baseline is None:
            continue
        for provider in providers:
            if provider == "torch-compile":
                continue
            result = next(
                (r for r in results if r["provider"] == provider and r[size_key] == size),
                None,
            )
            if result:
                speedup = baseline["median"] / result["median"]
                speedup_style = "green" if speedup > 1.0 else "red"
                speedup_table.add_row(
                    str(size),
                    provider,
                    f"[{speedup_style}]{speedup:.2f}x[/{speedup_style}]",
                )

    console.print(speedup_table)


def print_benchmark_progress(kind: str = "speed", **values: Any) -> None:
    """Print a compact progress line after an individual benchmark point completes."""
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    console.print(f"[dim]completed {kind}: {', '.join(parts)}[/dim]")


def print_speedup_table_2d(
    results: list[dict[str, Any]],
    providers: list[str],
    row_key: str,
    row_values: list[int],
    col_key: str,
    col_values: list[int],
) -> None:
    """Print provider speedups against torch-compile for each 2D sweep point."""
    if len(providers) <= 1 or not results:
        return

    console.print()
    speedup_table = Table(
        title="Speedup Summary (vs torch-compile baseline)",
        show_header=True,
        header_style="bold magenta",
    )
    speedup_table.add_column(row_key, justify="right")
    speedup_table.add_column(col_key, justify="right")
    speedup_table.add_column("Provider", style="cyan")
    speedup_table.add_column("Speedup", justify="right", style="green")

    for row_value in row_values:
        for col_value in col_values:
            baseline = next(
                (
                    r
                    for r in results
                    if r["provider"] == "torch-compile"
                    and r[row_key] == row_value
                    and r[col_key] == col_value
                ),
                None,
            )
            if baseline is None:
                continue
            for provider in providers:
                if provider == "torch-compile":
                    continue
                result = next(
                    (
                        r
                        for r in results
                        if r["provider"] == provider
                        and r[row_key] == row_value
                        and r[col_key] == col_value
                    ),
                    None,
                )
                if result:
                    speedup = baseline["median"] / result["median"]
                    speedup_style = "green" if speedup > 1.0 else "red"
                    speedup_table.add_row(
                        str(row_value),
                        str(col_value),
                        provider,
                        f"[{speedup_style}]{speedup:.2f}x[/{speedup_style}]",
                    )

    console.print(speedup_table)


PROVIDER_COLORS: dict[str, str] = {
    "greyhound": "#4285F4",
    "greyhound + quack": "#4285F4",
    "gram-newton-schulz": "#00ACC1",
    "torch-eager": "#EA4335",
    "torch.cross_entropy": "#EA4335",
    "torch-compile": "#FF7043",
    "liger": "#FBBC04",
    "quack": "#34A853",
    "fla": "#AB47BC",
    "causal-conv1d": "#F06292",
}


def save_plot(
    results: list[dict[str, Any]],
    providers: list[str],
    size_key: str,
    sizes: list[int],
    y_key: str = "gbps",
    title: str = "Benchmark",
    y_label: str = "GB/s",
    x_label: str | None = None,
    output_path: str | Path = "benchmark.png",
) -> None:
    """Save a line plot of benchmark results to a file.

    Args:
        results: List of result dicts from benchmarking.
        providers: Provider names to include.
        size_key: Key in results for the x-axis dimension (e.g. "d_model", "hidden_size").
        sizes: The x-axis values.
        y_key: Key in results for the y-axis metric (default: "gbps").
        title: Plot title.
        y_label: Y-axis label.
        x_label: X-axis label (defaults to size_key).
        output_path: File path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print(
            "[yellow]Warning: matplotlib not installed, skipping plot. "
            "Install with: pip install matplotlib[/yellow]"
        )
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for provider in providers:
        provider_results = [r for r in results if r["provider"] == provider]
        xs = [r[size_key] for r in provider_results]
        ys = [r[y_key] for r in provider_results]
        color = PROVIDER_COLORS.get(provider)
        ax.plot(xs, ys, marker="o", label=provider, color=color, linewidth=2, markersize=6)

    ax.set_xlabel(x_label or size_key, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    console.print(f"[green]Plot saved to {output_path}[/green]")


def save_heatmap(
    results: list[dict[str, Any]],
    providers: list[str],
    row_key: str,
    row_values: list[int],
    col_key: str,
    col_values: list[int],
    y_key: str,
    title: str,
    y_label: str,
    output_path: str | Path,
) -> None:
    """Save one heatmap per provider for a 2D benchmark sweep."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        console.print(
            "[yellow]Warning: matplotlib/numpy not installed, skipping plot. "
            "Install with: pip install matplotlib numpy[/yellow]"
        )
        return

    n_providers = len(providers)
    fig_width = max(6, 4 * n_providers)
    fig, axes = plt.subplots(1, n_providers, figsize=(fig_width, 5), squeeze=False)

    values_by_provider: dict[str, Any] = {}
    finite_values: list[float] = []
    for provider in providers:
        grid = np.full((len(row_values), len(col_values)), np.nan, dtype=float)
        for row_idx, row_value in enumerate(row_values):
            for col_idx, col_value in enumerate(col_values):
                result = next(
                    (
                        r
                        for r in results
                        if r["provider"] == provider
                        and r[row_key] == row_value
                        and r[col_key] == col_value
                    ),
                    None,
                )
                if result is not None:
                    grid[row_idx, col_idx] = result[y_key]
        finite_values.extend(grid[np.isfinite(grid)].tolist())
        values_by_provider[provider] = grid

    vmin = min(finite_values) if finite_values else None
    vmax = max(finite_values) if finite_values else None
    image = None
    for ax, provider in zip(axes[0], providers):
        grid = values_by_provider[provider]
        image = ax.imshow(grid, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(provider)
        ax.set_xlabel(col_key)
        ax.set_ylabel(row_key)
        ax.set_xticks(range(len(col_values)))
        ax.set_xticklabels([str(v) for v in col_values], rotation=45, ha="right")
        ax.set_yticks(range(len(row_values)))
        ax.set_yticklabels([str(v) for v in row_values])
        for row_idx in range(len(row_values)):
            for col_idx in range(len(col_values)):
                value = grid[row_idx, col_idx]
                if np.isfinite(value):
                    ax.text(col_idx, row_idx, f"{value:.2g}", ha="center", va="center")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=y_label)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    console.print(f"[green]Plot saved to {output_path}[/green]")


def get_gpu_name() -> str:
    """Return the name of the current CUDA device."""
    return torch.cuda.get_device_name()


def get_git_commit() -> str:
    """Return the short git commit hash of the current HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def save_results_csv(rows: list[dict[str, Any]], csv_path: Path = BENCHMARK_CSV) -> None:
    """Merge benchmark result rows into the CSV file.

    Each row is a dict with keys matching CSV_COLUMNS. Rows are keyed by
    CSV_KEY_COLUMNS — existing rows with the same key are overwritten,
    new keys are appended. The file is written back sorted by key columns.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[tuple[str, ...], dict[str, Any]] = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(str(row.get(c, "")) for c in CSV_KEY_COLUMNS)
                existing[key] = {c: row.get(c, "") for c in CSV_COLUMNS}

    for row in rows:
        key = tuple(str(row.get(c, "")) for c in CSV_KEY_COLUMNS)
        existing[key] = {c: row.get(c, "") for c in CSV_COLUMNS}

    sorted_rows = sorted(
        existing.values(), key=lambda r: tuple(str(r.get(c, "")) for c in CSV_KEY_COLUMNS)
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted_rows)

    console.print(f"[green]Benchmark results saved to {csv_path} ({len(sorted_rows)} rows)[/green]")
