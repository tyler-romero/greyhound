"""Generate benchmark plots from benchmark_data.csv.

Reads the CSV written by benchmark scripts and produces one plot per unique
combination of (kernel_name, operation_mode, metric_name, extra_config, gpu_name).
Plots are written to docs/public/assets/plots/ and docs/public/assets/plots_html/ by default
so newly generated plots are published by the docs site.

Usage:
    uv run python src/benchmarks/plot_from_csv.py
    uv run python src/benchmarks/plot_from_csv.py --csv path/to/data.csv
    uv run python src/benchmarks/plot_from_csv.py --output-dir path/to/plots
    uv run python src/benchmarks/plot_from_csv.py --html-output-dir path/to/html
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROVIDER_COLORS: dict[str, str] = {
    "greyhound": "#4285F4",
    "greyhound + quack": "#4285F4",
    "gram-newton-schulz": "#00ACC1",
    "dion": "#7E57C2",
    "torch-eager": "#EA4335",
    "torch.cross_entropy": "#EA4335",
    "torch-compile": "#FF7043",
    "liger": "#FBBC04",
    "quack": "#34A853",
    "fla": "#AB47BC",
    "causal-conv1d": "#F06292",
}

PROVIDER_ORDER = list(PROVIDER_COLORS.keys())

DEFAULT_CSV = Path(__file__).parent / "data" / "benchmark_data.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "public" / "assets" / "plots"
DEFAULT_HTML_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "public" / "assets" / "plots_html"
)

METRIC_LABELS: dict[str, str] = {
    "speed": "Median Time ({unit})",
    "memory": "Peak Memory ({unit})",
    "gbps": "Memory Bandwidth ({unit})",
    "tflops": "TFLOPS ({unit})",
}


def slugify(s: str) -> str:
    return s.replace(" ", "_").replace(",", "").replace('"', "").replace(":", "-")


def format_config(config_str: str) -> str:
    """Turn a JSON config string into a readable subtitle."""
    try:
        cfg = json.loads(config_str)
    except (json.JSONDecodeError, TypeError):
        return config_str
    parts = [f"{k}={v}" for k, v in sorted(cfg.items())]
    return ", ".join(parts)


def parse_csv(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


GroupKey = tuple[str, str, str, str, str]
TradeoffGroupKey = tuple[str, str, str, str, str]


def group_rows(rows: list[dict[str, str]]) -> dict[GroupKey, list[dict[str, str]]]:
    """Group rows by (kernel_name, operation_mode, metric_name, extra_config, hardware)."""
    groups: dict[GroupKey, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        gpu_name = row.get("gpu_name", "").strip() or "Unknown GPU"
        key: GroupKey = (
            row["kernel_name"],
            row["operation_mode"],
            row["metric_name"],
            row["extra_benchmark_config_str"],
            gpu_name,
        )
        groups[key] = groups.get(key, [])
        groups[key].append(row)
    return groups


def group_tradeoff_rows(rows: list[dict[str, str]]) -> dict[TradeoffGroupKey, list[dict[str, str]]]:
    """Group rows for derived speed-vs-memory plots."""
    groups: dict[TradeoffGroupKey, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["metric_name"] not in {"memory", "speed"}:
            continue
        gpu_name = row.get("gpu_name", "").strip() or "Unknown GPU"
        key: TradeoffGroupKey = (
            row["kernel_name"],
            row["operation_mode"],
            row["extra_benchmark_config_str"],
            gpu_name,
            row["x_name"],
        )
        groups[key].append(row)
    return groups


def _finite_y_value(row: dict[str, str]) -> float | None:
    value = row["y_value_50"]
    if not value or value == "nan":
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def make_plot(
    group_key: GroupKey,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> Path | None:
    kernel_name, op_mode, metric_name, config_str, gpu_name = group_key
    x_name = rows[0]["x_name"]
    if "," in x_name:
        return None

    by_provider: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_provider[row["kernel_provider"]].append(row)

    has_data = False
    for provider_rows in by_provider.values():
        for row in provider_rows:
            if _finite_y_value(row) is not None:
                has_data = True
                break
        if has_data:
            break
    if not has_data:
        return None

    sorted_providers = sorted(
        by_provider.keys(),
        key=lambda p: PROVIDER_ORDER.index(p) if p in PROVIDER_ORDER else len(PROVIDER_ORDER),
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    all_x_values: set[float] = set()
    for provider in sorted_providers:
        provider_rows = sorted(by_provider[provider], key=lambda r: float(r["x_value"]))
        xs: list[float] = []
        ys: list[float] = []
        low: list[float] = []
        high: list[float] = []
        for row in provider_rows:
            y = _finite_y_value(row)
            if y is None:
                continue
            x = float(row["x_value"])
            xs.append(x)
            ys.append(y)
            y20 = row.get("y_value_20")
            y80 = row.get("y_value_80")
            if y20 and y20 != "nan":
                low.append(float(y20))
            else:
                low.append(y)
            if y80 and y80 != "nan":
                high.append(float(y80))
            else:
                high.append(y)
            all_x_values.add(x)

        if not xs:
            continue

        color = PROVIDER_COLORS.get(provider)
        if low != ys or high != ys:
            ax.fill_between(xs, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(
            xs,
            ys,
            marker="o",
            label=provider,
            color=color,
            linewidth=2.4,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )

    if not all_x_values:
        plt.close(fig)
        return None

    metric_unit = rows[0]["metric_unit"]

    y_label = METRIC_LABELS.get(metric_name, "{unit}").format(unit=metric_unit)

    config_label = format_config(config_str)
    title = f"{kernel_name} ({op_mode}, {metric_name})"
    hardware_label = f"Hardware: {gpu_name}"

    ax.set_xlabel(x_name, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if config_label:
        ax.text(
            0.5,
            1.0,
            config_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="gray",
        )
    ax.text(
        0.5,
        0.95,
        hardware_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5563",
    )
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35, color="#9ca3af")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sorted_x = sorted(all_x_values)
    ax.set_xticks(sorted_x)
    ax.set_xticklabels(
        [f"{int(x):,}" if x == int(x) else f"{x:g}" for x in sorted_x],
        rotation=35,
        ha="right",
    )
    ax.tick_params(axis="both", labelsize=10.5)
    fig.tight_layout()

    filename = slugify(f"{kernel_name}_{op_mode}_{metric_name}_{config_label}_{gpu_name}") + ".png"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _x_value_sort_key(x_value: str) -> tuple[int, tuple[float, ...] | str]:
    try:
        return (0, tuple(float(part.strip()) for part in x_value.split(",")))
    except ValueError:
        return (1, x_value)


def _is_pareto_point(
    point: dict[str, float | str],
    candidates: list[dict[str, float | str]],
) -> bool:
    memory = float(point["memory"])
    speed = float(point["speed"])
    for candidate in candidates:
        if candidate is point:
            continue
        candidate_memory = float(candidate["memory"])
        candidate_speed = float(candidate["speed"])
        if (
            candidate_memory <= memory
            and candidate_speed <= speed
            and (candidate_memory < memory or candidate_speed < speed)
        ):
            return False
    return True


def make_memory_speed_pareto_plot(
    group_key: TradeoffGroupKey,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> Path | None:
    kernel_name, op_mode, config_str, gpu_name, x_name = group_key

    speed_rows: dict[tuple[str, str], dict[str, str]] = {}
    memory_rows: dict[tuple[str, str], dict[str, str]] = {}
    speed_unit = None
    memory_unit = None
    for row in rows:
        key = (row["kernel_provider"], row["x_value"])
        if row["metric_name"] == "speed":
            speed_rows[key] = row
            speed_unit = speed_unit or row["metric_unit"]
        elif row["metric_name"] == "memory":
            memory_rows[key] = row
            memory_unit = memory_unit or row["metric_unit"]

    points: list[dict[str, float | str]] = []
    for key in sorted(
        speed_rows.keys() & memory_rows.keys(), key=lambda k: (k[0], _x_value_sort_key(k[1]))
    ):
        provider, x_value = key
        speed = _finite_y_value(speed_rows[key])
        memory = _finite_y_value(memory_rows[key])
        if speed is None or memory is None:
            continue
        points.append(
            {
                "provider": provider,
                "x_value": x_value,
                "speed": speed,
                "memory": memory,
            }
        )

    if not points or len({point["provider"] for point in points}) < 2:
        return None

    points_by_x_value: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for point in points:
        points_by_x_value[str(point["x_value"])].append(point)

    frontier_points: set[tuple[str, str]] = set()
    frontier_lines: list[list[dict[str, float | str]]] = []
    for x_value in sorted(points_by_x_value, key=_x_value_sort_key):
        candidates = points_by_x_value[x_value]
        if len(candidates) < 2:
            continue
        frontier = [point for point in candidates if _is_pareto_point(point, candidates)]
        if not frontier:
            continue
        frontier = sorted(
            frontier, key=lambda point: (float(point["memory"]), float(point["speed"]))
        )
        frontier_lines.append(frontier)
        for point in frontier:
            frontier_points.add((str(point["provider"]), str(point["x_value"])))

    if not frontier_points:
        return None

    sorted_providers = sorted(
        {str(point["provider"]) for point in points},
        key=lambda p: PROVIDER_ORDER.index(p) if p in PROVIDER_ORDER else len(PROVIDER_ORDER),
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    for provider in sorted_providers:
        provider_points = [point for point in points if point["provider"] == provider]
        provider_frontier_points = [
            point
            for point in provider_points
            if (str(point["provider"]), str(point["x_value"])) in frontier_points
        ]
        provider_dominated_points = [
            point
            for point in provider_points
            if (str(point["provider"]), str(point["x_value"])) not in frontier_points
        ]
        color = PROVIDER_COLORS.get(provider)
        if provider_dominated_points:
            ax.scatter(
                [float(point["memory"]) for point in provider_dominated_points],
                [float(point["speed"]) for point in provider_dominated_points],
                color=color,
                edgecolors="white",
                linewidths=0.6,
                s=42,
                alpha=0.32,
                label=None if provider_frontier_points else provider,
            )
        if provider_frontier_points:
            ax.scatter(
                [float(point["memory"]) for point in provider_frontier_points],
                [float(point["speed"]) for point in provider_frontier_points],
                color=color,
                edgecolors="#111827",
                linewidths=1.0,
                s=78,
                alpha=0.92,
                label=provider,
            )

    for frontier in frontier_lines:
        if len(frontier) < 2:
            continue
        ax.plot(
            [float(point["memory"]) for point in frontier],
            [float(point["speed"]) for point in frontier],
            color="#111827",
            linewidth=1.2,
            alpha=0.22,
            zorder=1,
        )

    speed_label = METRIC_LABELS["speed"].format(unit=speed_unit or "")
    memory_label = METRIC_LABELS["memory"].format(unit=memory_unit or "")
    config_label = format_config(config_str)
    title = f"{kernel_name} ({op_mode}, memory-speed Pareto)"
    subtitle = f"Frontier is computed within each {x_name} point. Lower-left is better."

    ax.set_xlabel(memory_label, fontsize=12)
    ax.set_ylabel(speed_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if config_label:
        ax.text(
            0.5,
            1.0,
            config_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="gray",
        )
    ax.text(
        0.5,
        0.94,
        f"{subtitle}\nHardware: {gpu_name}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5563",
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#9ca3af",
            markeredgecolor="#111827",
            markersize=8,
            label="Pareto point",
        )
    )
    labels.append("Pareto point")
    ax.legend(handles, labels, fontsize=10, frameon=False, ncol=2, loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35, color="#9ca3af")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5)
    fig.tight_layout()

    filename = (
        slugify(f"{kernel_name}_{op_mode}_memory_speed_pareto_{config_label}_{gpu_name}") + ".png"
    )
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_2d_x_value(row: dict[str, str]) -> tuple[float, float]:
    left, right = row["x_value"].split(",")
    return float(left), float(right)


def _parse_numeric_x_values(row: dict[str, str]) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in row["x_value"].split(","))


def _make_plotly_html_iframe_friendly(output_path: Path) -> None:
    html = output_path.read_text()
    styles = """
<style>
  html,
  body {
    width: 100%;
    height: 100%;
    margin: 0;
    overflow: hidden;
  }

  .plotly-graph-div {
    width: 100% !important;
    height: 100% !important;
  }
</style>
"""
    resize_script = """
<script>
  (() => {
    const resizePlot = () => {
      const graph = document.querySelector(".plotly-graph-div");
      if (graph && window.Plotly) {
        window.Plotly.Plots.resize(graph);
      }
    };

    window.addEventListener("load", resizePlot);
    window.addEventListener("resize", resizePlot);
  })();
</script>
"""
    if "<head>" in html:
        html = html.replace("<head>", f"<head>{styles}", 1)
    else:
        html = f"{styles}{html}"
    if "</body>" in html:
        html = html.replace("</body>", f"{resize_script}</body>", 1)
    else:
        html = f"{html}{resize_script}"
    output_path.write_text(html)


def make_plotly_3d_plot(
    group_key: GroupKey,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> Path | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed; skipping HTML 3D plot generation.")
        return None

    kernel_name, op_mode, metric_name, config_str, gpu_name = group_key
    x_names = [part.strip() for part in rows[0]["x_name"].split(",")]
    if len(x_names) == 2:
        selector_name = None
        selector_values = [None]
        row_name, col_name = x_names
        parsed_rows = [(row, None, *_parse_2d_x_value(row)) for row in rows]
    elif len(x_names) == 3:
        selector_name, row_name, col_name = x_names
        parsed_rows = []
        for row in rows:
            selector_value, row_value, col_value = _parse_numeric_x_values(row)
            parsed_rows.append((row, selector_value, row_value, col_value))
        selector_values = sorted({selector_value for _, selector_value, _, _ in parsed_rows})
    else:
        print(
            f"Skipping HTML 3D plot for {kernel_name}: expected 2 or 3 x dimensions, "
            f"got {rows[0]['x_name']!r}."
        )
        return None

    row_values = sorted({row_value for _, _, row_value, _ in parsed_rows})
    col_values = sorted({col_value for _, _, _, col_value in parsed_rows})

    by_provider: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_provider[row["kernel_provider"]].append(row)

    sorted_providers = sorted(
        by_provider.keys(),
        key=lambda p: PROVIDER_ORDER.index(p) if p in PROVIDER_ORDER else len(PROVIDER_ORDER),
    )

    parsed_by_row_id = {
        id(row): (selector_value, row_value, col_value)
        for row, selector_value, row_value, col_value in parsed_rows
    }
    values_by_provider: dict[str, dict[tuple[float | None, float, float], float]] = {}
    has_data = False
    for provider in sorted_providers:
        values: dict[tuple[float | None, float, float], float] = {}
        for row in by_provider[provider]:
            y_val = row["y_value_50"]
            if not y_val or y_val == "nan":
                continue
            y = float(y_val)
            if math.isnan(y):
                continue
            selector_value, row_value, col_value = parsed_by_row_id[id(row)]
            values[(selector_value, row_value, col_value)] = y
            has_data = True
        values_by_provider[provider] = values

    if not has_data:
        return None

    metric_unit = rows[0]["metric_unit"]
    y_label = METRIC_LABELS.get(metric_name, "{unit}").format(unit=metric_unit)
    config_label = format_config(config_str)
    title = f"{kernel_name} ({op_mode}, {metric_name})"
    first_selector_value = selector_values[0]
    subtitle = (
        f"{config_label}<br>Hardware: {gpu_name}" if config_label else f"Hardware: {gpu_name}"
    )

    fig = go.Figure()
    trace_selector_values: list[float | None] = []
    for provider in sorted_providers:
        provider_values = values_by_provider[provider]
        color = PROVIDER_COLORS.get(provider)
        for selector_value in selector_values:
            for row_idx, row_value in enumerate(row_values):
                points = [
                    (col_value, provider_values[(selector_value, row_value, col_value)])
                    for col_value in col_values
                    if (selector_value, row_value, col_value) in provider_values
                ]
                if not points:
                    continue
                xs = [row_value for _ in points]
                ys = [col_value for col_value, _ in points]
                zs = [metric_value for _, metric_value in points]
                trace_selector_values.append(selector_value)
                selector_hover = (
                    "" if selector_name is None else f"{selector_name}: {selector_value:g}<br>"
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="markers",
                        name=provider,
                        legendgroup=provider,
                        showlegend=row_idx == 0,
                        visible=selector_value == first_selector_value,
                        marker={"color": color, "size": 4.5},
                        hovertemplate=(
                            f"{provider}<br>"
                            f"{selector_hover}"
                            f"{row_name}: %{{x}}<br>"
                            f"{col_name}: %{{y}}<br>"
                            f"{y_label}: %{{z:.4g}}"
                            "<extra></extra>"
                        ),
                    )
                )

    updatemenus = []
    if selector_name is not None:
        buttons = []
        for selector_value in selector_values:
            visible = [
                trace_selector_value == selector_value
                for trace_selector_value in trace_selector_values
            ]
            buttons.append(
                {
                    "label": f"{selector_name}={selector_value:g}",
                    "method": "update",
                    "args": [
                        {"visible": visible},
                        {
                            "title": (
                                f"{title}<br><sup>{selector_name}={selector_value:g} | "
                                f"{subtitle}</sup>"
                            )
                        },
                    ],
                }
            )
        updatemenus.append(
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.98,
                "xanchor": "right",
                "y": 1.08,
                "yanchor": "top",
            }
        )

    title_prefix = (
        title
        if selector_name is None
        else f"{title}<br><sup>{selector_name}={first_selector_value:g} | {subtitle}</sup>"
    )
    title_text = f"{title}<br><sup>{subtitle}</sup>" if selector_name is None else title_prefix
    fig.update_layout(
        title=title_text,
        scene={
            "xaxis_title": row_name,
            "yaxis_title": col_name,
            "zaxis_title": y_label,
        },
        updatemenus=updatemenus,
        autosize=True,
        height=720,
        margin={"l": 0, "r": 0, "t": 92, "b": 0},
        legend={
            "bgcolor": "rgba(255,255,255,0.75)",
            "itemsizing": "constant",
            "x": 0.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
    )

    filename = slugify(f"{kernel_name}_{op_mode}_{metric_name}_{config_label}_{gpu_name}") + ".html"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )
    _make_plotly_html_iframe_friendly(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from benchmark CSV data.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to benchmark CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output plots"
    )
    parser.add_argument(
        "--html-output-dir",
        type=Path,
        default=DEFAULT_HTML_OUTPUT_DIR,
        help="Directory for interactive Plotly HTML 3D plots for 2D sweep rows",
    )
    args = parser.parse_args()

    rows = parse_csv(args.csv)
    groups = group_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for group_key, group_rows_list in sorted(groups.items()):
        result = make_plot(group_key, group_rows_list, args.output_dir)
        if result:
            print(f"  {result}")
            count += 1

    print(f"Generated {count} plots in {args.output_dir}")

    pareto_count = 0
    for group_key, group_rows_list in sorted(group_tradeoff_rows(rows).items()):
        result = make_memory_speed_pareto_plot(group_key, group_rows_list, args.output_dir)
        if result:
            print(f"  {result}")
            pareto_count += 1

    print(f"Generated {pareto_count} memory-speed Pareto plots in {args.output_dir}")

    if args.html_output_dir is not None:
        args.html_output_dir.mkdir(parents=True, exist_ok=True)
        html_count = 0
        for group_key, group_rows_list in sorted(groups.items()):
            if "," not in group_rows_list[0]["x_name"]:
                continue
            result = make_plotly_3d_plot(group_key, group_rows_list, args.html_output_dir)
            if result:
                print(f"  {result}")
                html_count += 1

        print(f"Generated {html_count} HTML plots in {args.html_output_dir}")


if __name__ == "__main__":
    main()
