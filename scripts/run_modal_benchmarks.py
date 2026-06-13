#!/usr/bin/env python3
"""Run greyhound benchmark scripts on Modal and merge the CSV results.

Examples:
    uv run --extra modal python scripts/run_modal_benchmarks.py \
        --benchmark "src/benchmarks/causal_conv1d_bench.py --mode full"

    uv run --extra modal python scripts/run_modal_benchmarks.py \
        --gpu H100 \
        --extra quack \
        --benchmark "src/benchmarks/chunked_linear_cross_entropy_bench.py --providers quack --mode full"

    uv run --extra modal python scripts/run_modal_benchmarks.py \
        --gpu H100 \
        --benchmark "src/benchmarks/chunked_linear_cross_entropy_bench.py --mode forward" \
        --benchmark "src/benchmarks/logprobs_bench.py --mode forward"
"""

import csv
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_CSV = SRC_DIR / "benchmarks" / "data" / "benchmark_data.csv"
DEFAULT_REMOTE_CSV = "/tmp/greyhound_modal_benchmark_data.csv"
DEFAULT_EXTRAS = ("modal",)
DEFAULT_GROUPS = ("thirdparty",)
AVAILABLE_EXTRAS = ("modal", "quack")
AVAILABLE_GROUPS = ("thirdparty",)
DEFAULT_BENCHMARKS = [
    "src/benchmarks/causal_conv1d_bench.py",
    "src/benchmarks/cross_entropy_with_grad_bench.py",
    "src/benchmarks/chunked_linear_cross_entropy_bench.py",
    "src/benchmarks/logprobs_bench.py",
    "src/benchmarks/newton_schulz_bench.py",
]


def normalize_benchmark_commands(raw_commands: tuple[str, ...]) -> list[list[str]]:
    commands = raw_commands or DEFAULT_BENCHMARKS
    parsed: list[list[str]] = []
    for command in commands:
        parts = shlex.split(command)
        if not parts:
            raise ValueError("Empty benchmark command")
        script = Path(parts[0])
        if script.is_absolute():
            try:
                script = script.resolve().relative_to(REPO_ROOT)
            except ValueError as exc:
                raise ValueError(
                    f"Benchmark script must be inside {REPO_ROOT}: {parts[0]}"
                ) from exc
        if not (REPO_ROOT / script).exists():
            raise FileNotFoundError(f"Benchmark script does not exist: {script}")
        parsed.append([script.as_posix(), *parts[1:]])
    return parsed


def normalize_extras(raw_extras: tuple[str, ...], skip_thirdparty: bool) -> list[str]:
    if raw_extras and skip_thirdparty:
        raise click.UsageError("Use either --extra or --skip-thirdparty, not both.")

    selected = ["modal"] if skip_thirdparty else list(raw_extras or DEFAULT_EXTRAS)

    if "modal" not in selected:
        selected.insert(0, "modal")

    deduped = []
    for extra in selected:
        if extra not in deduped:
            deduped.append(extra)
    return deduped


def normalize_groups(raw_groups: tuple[str, ...], skip_thirdparty: bool) -> list[str]:
    if raw_groups and skip_thirdparty:
        raise click.UsageError("Use either --group or --skip-thirdparty, not both.")

    selected = [] if skip_thirdparty else list(raw_groups or DEFAULT_GROUPS)
    deduped = []
    for group in selected:
        if group not in deduped:
            deduped.append(group)
    return deduped


def build_modal_image(
    extras: list[str],
    groups: list[str],
    python_version: str,
    base_image: str,
) -> Any:
    modal = import_modal()

    ignore = [
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".astro",
        "dist",
        "docs/node_modules",
    ]
    if base_image == "cuda-devel":
        image = modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu24.04",
            add_python=python_version,
        ).apt_install("build-essential", "g++", "gcc", "git", "ninja-build")
    else:
        image = modal.Image.debian_slim(python_version=python_version).apt_install("git")

    return (
        image.env({"CC": "gcc", "CXX": "g++"})
        .uv_sync(extras=extras, groups=groups)
        .add_local_dir(REPO_ROOT, remote_path="/root/greyhound", ignore=ignore)
    )


def import_modal() -> Any:
    try:
        import modal
    except ImportError as exc:
        raise SystemExit(
            "Modal is not installed in this environment. Run with "
            "`uv run --extra modal python scripts/run_modal_benchmarks.py ...` "
            "or install the `modal` optional extra first."
        ) from exc
    return modal


def merge_csv_rows(rows: list[dict[str, str]], output_csv: Path) -> None:
    sys.path.insert(0, str(SRC_DIR))
    from benchmarks.utils import save_results_csv

    save_results_csv(rows, output_csv)


def parse_csv_text(csv_text: str) -> list[dict[str, str]]:
    if not csv_text.strip():
        return []
    return list(csv.DictReader(csv_text.splitlines()))


@click.command(help="Run benchmark scripts on Modal, fetch their CSV output, and merge it locally.")
@click.option(
    "-b",
    "--benchmark",
    multiple=True,
    help=(
        "Benchmark command to run, including script path and any script args. May be repeated. "
        "Defaults to all known benchmark scripts with default args."
    ),
)
@click.option(
    "--gpu", default="H100", show_default=True, help="Modal GPU type, e.g. A100, H100, B200."
)
@click.option("--timeout", default=60 * 60, show_default=True, help="Remote timeout in seconds.")
@click.option(
    "--python-version",
    default="3.12",
    show_default=True,
    help="Python version for the Modal image.",
)
@click.option(
    "--base-image",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "slim", "cuda-devel"]),
    help="Modal base image. auto uses a CUDA devel image when provider extras are enabled.",
)
@click.option(
    "--torch-compile-mode",
    default="default",
    show_default=True,
    help=(
        "Value for GREYHOUND_TORCH_COMPILE_MODE inside Modal. Use default for normal "
        "torch.compile, none to disable compile, or a torch.compile mode such as "
        "max-autotune-no-cudagraphs."
    ),
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=DEFAULT_CSV,
    show_default=True,
    help="Local benchmark CSV to merge remote results into.",
)
@click.option(
    "--remote-csv",
    default=DEFAULT_REMOTE_CSV,
    show_default=True,
    help="Temporary CSV path used inside the Modal container.",
)
@click.option(
    "--extra",
    "extras",
    multiple=True,
    type=click.Choice(AVAILABLE_EXTRAS),
    help=(
        "Project optional dependency extra to install in the Modal image. May be repeated. "
        "Defaults to modal. The modal extra is added automatically."
    ),
)
@click.option(
    "--group",
    "groups",
    multiple=True,
    type=click.Choice(AVAILABLE_GROUPS),
    help=(
        "Project dependency group to install in the Modal image. May be repeated. "
        "Defaults to thirdparty."
    ),
)
@click.option(
    "--skip-thirdparty",
    is_flag=True,
    help="Compatibility shortcut for installing only the modal extra.",
)
def main(
    benchmark: tuple[str, ...],
    gpu: str,
    timeout: int,
    python_version: str,
    base_image: str,
    torch_compile_mode: str,
    output_csv: Path,
    remote_csv: str,
    extras: tuple[str, ...],
    groups: tuple[str, ...],
    skip_thirdparty: bool,
) -> None:
    benchmark_commands = normalize_benchmark_commands(benchmark)
    selected_extras = normalize_extras(extras, skip_thirdparty)
    selected_groups = normalize_groups(groups, skip_thirdparty)
    if base_image == "auto":
        base_image = (
            "slim" if selected_extras == ["modal"] and not selected_groups else "cuda-devel"
        )

    modal = import_modal()
    app = modal.App("greyhound-modal-benchmarks")
    image = build_modal_image(
        extras=selected_extras,
        groups=selected_groups,
        python_version=python_version,
        base_image=base_image,
    )

    @app.function(image=image, gpu=gpu, timeout=timeout, serialized=True)
    def run_remote_benchmarks(
        commands: list[list[str]],
        remote_csv: str,
        torch_compile_mode: str,
    ) -> dict[str, str]:
        import os
        import shlex
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path("/root/greyhound")
        csv_path = Path(remote_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.unlink(missing_ok=True)

        env = {
            **os.environ,
            "PYTHONPATH": str(repo_root / "src"),
            "GREYHOUND_BENCHMARK_CSV": str(csv_path),
            "GREYHOUND_TORCH_COMPILE_MODE": torch_compile_mode,
        }

        rendered_commands = []
        for command in commands:
            script = repo_root / command[0]
            wrapper = (
                "import runpy, sys; "
                "import torch._functorch.config as c; "
                "c.donated_buffer = False; "
                "sys.argv = [sys.argv[1], *sys.argv[2:]]; "
                "runpy.run_path(sys.argv[0], run_name='__main__')"
            )
            argv = [sys.executable, "-c", wrapper, str(script), *command[1:]]
            rendered_commands.append(" ".join(shlex.quote(part) for part in argv))
            subprocess.run(argv, cwd=repo_root, env=env, check=True)

        return {
            "commands": "\n".join(rendered_commands),
            "csv": csv_path.read_text() if csv_path.exists() else "",
        }

    with modal.enable_output():
        with app.run():
            result = run_remote_benchmarks.remote(
                benchmark_commands, remote_csv, torch_compile_mode
            )

    rows = parse_csv_text(result["csv"])
    if not rows:
        raise RuntimeError("Modal benchmarks completed without producing any CSV rows")

    merge_csv_rows(rows, output_csv)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(result["csv"])
        remote_csv_copy = Path(f.name)

    print(f"Ran {len(benchmark_commands)} benchmark command(s) on Modal {gpu}.")
    print(f"Fetched {len(rows)} CSV row(s) from Modal.")
    print(f"Merged results into {output_csv}.")
    print(f"Saved a copy of the remote CSV at {remote_csv_copy}.")


if __name__ == "__main__":
    main()
