from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EVAL_FILE_RE = re.compile(
    r"(?P<model>.+?)_train-(?P<train_generator>[^_]+)_comp-(?P<train_compression>[^_]+)"
    r"_eval-(?P<eval_generator>[^_]+)_test-(?P<test_compression>[^_]+)"
    r"_(?P<timestamp>\d{8}_\d{6})_metrics\.json$"
)

GENERATOR_ALIASES = {
    "adml": "adm",
}

DEFAULT_GENERATORS = ["biggan", "adm", "glide"]


@dataclass(frozen=True)
class EvalResult:
    path: Path
    train_generator: str
    train_compression: str
    eval_generator: str
    test_compression: str
    timestamp: str
    metrics: dict[str, Any]
    preprocess_mode: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3x3 heatmap from outputs/eval metric JSON files.",
    )
    parser.add_argument("--eval-dir", default="outputs/eval", help="Directory containing *_metrics.json files.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to outputs/plots/...")
    parser.add_argument("--train-compression", required=True, help="Training compression level, e.g. raw, jpeg95.")
    parser.add_argument("--test-compression", required=True, help="Test compression level, e.g. raw, jpeg95.")
    parser.add_argument("--metric", required=True, help="Metric key to plot, e.g. accuracy, recall, roc_auc, loss.")
    parser.add_argument(
        "--preprocess-mode",
        default=None,
        help="Optional filter for eval JSON payload preprocess_mode, e.g. standard or size-constrained.",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=DEFAULT_GENERATORS,
        help="Generator order for both axes. Defaults to: biggan adm glide.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--cmap", default="Blues", help="Matplotlib colormap name.")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum after scaling.")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum after scaling.")
    parser.add_argument(
        "--scale",
        choices=["auto", "raw", "percent"],
        default="auto",
        help="Use raw values, percentages, or auto-percent for metrics in [0, 1].",
    )
    parser.add_argument("--annot-format", default=".2f", help="Cell annotation format.")
    parser.add_argument("--dpi", type=int, default=200, help="Saved image DPI.")
    parser.add_argument("--show-sources", action="store_true", help="Print the JSON file used for each cell.")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    return GENERATOR_ALIASES.get(normalized, normalized)


def display_name(name: str) -> str:
    mapping = {
        "adm": "ADM",
        "biggan": "BigGAN",
        "glide": "GLIDE",
    }
    return mapping.get(name, name.upper())


def load_results(eval_dir: Path) -> list[EvalResult]:
    results: list[EvalResult] = []
    for path in eval_dir.glob("*_metrics.json"):
        match = EVAL_FILE_RE.match(path.name)
        if not match:
            continue

        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        groups = match.groupdict()
        results.append(
            EvalResult(
                path=path,
                train_generator=normalize_name(groups["train_generator"]),
                train_compression=groups["train_compression"].lower(),
                eval_generator=normalize_name(groups["eval_generator"]),
                test_compression=groups["test_compression"].lower(),
                timestamp=groups["timestamp"],
                metrics=metrics,
                preprocess_mode=metrics.get("preprocess_mode"),
            )
        )
    return results


def pick_latest_result(
    results: list[EvalResult],
    train_generator: str,
    eval_generator: str,
    train_compression: str,
    test_compression: str,
    preprocess_mode: str | None = None,
) -> EvalResult | None:
    matches = [
        result
        for result in results
        if result.train_generator == train_generator
        and result.eval_generator == eval_generator
        and result.train_compression == train_compression
        and result.test_compression == test_compression
        and (preprocess_mode is None or result.preprocess_mode == preprocess_mode)
    ]
    if not matches:
        return None
    return max(matches, key=lambda result: result.timestamp)


def build_matrix(
    results: list[EvalResult],
    generators: list[str],
    train_compression: str,
    test_compression: str,
    metric: str,
    preprocess_mode: str | None = None,
) -> tuple[np.ndarray, list[list[EvalResult | None]]]:
    matrix = np.full((len(generators), len(generators)), np.nan, dtype=float)
    sources: list[list[EvalResult | None]] = []
    missing_metric_files: list[Path] = []

    for row_idx, train_generator in enumerate(generators):
        source_row: list[EvalResult | None] = []
        for col_idx, eval_generator in enumerate(generators):
            result = pick_latest_result(
                results,
                train_generator=train_generator,
                eval_generator=eval_generator,
                train_compression=train_compression,
                test_compression=test_compression,
                preprocess_mode=preprocess_mode,
            )
            source_row.append(result)
            if result is None:
                continue

            value = result.metrics.get(metric)
            if value is None:
                missing_metric_files.append(result.path)
                continue
            matrix[row_idx, col_idx] = float(value)
        sources.append(source_row)

    if missing_metric_files:
        examples = ", ".join(str(path) for path in missing_metric_files[:3])
        raise KeyError(f"Metric '{metric}' was not found in matching file(s): {examples}")

    return matrix, sources


def should_scale_to_percent(matrix: np.ndarray, scale: str) -> bool:
    if scale == "percent":
        return True
    if scale == "raw":
        return False

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return False
    return float(np.nanmin(finite_values)) >= 0.0 and float(np.nanmax(finite_values)) <= 1.0


def default_output_path(metric: str, train_compression: str, test_compression: str) -> Path:
    return Path("outputs/plots") / f"eval_heatmap_{metric}_train-{train_compression}_test-{test_compression}.png"


def annotate_cells(ax: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    finite_values = matrix[np.isfinite(matrix)]
    threshold = float(np.nanmean(finite_values)) if finite_values.size else math.inf

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                text = format(value, fmt)
                color = "white" if value >= threshold else "#172033"
            else:
                text = "N/A"
                color = "#697386"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=9)


def plot_heatmap(
    matrix: np.ndarray,
    generators: list[str],
    metric: str,
    train_compression: str,
    test_compression: str,
    output_path: Path,
    title: str | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    annot_format: str,
    dpi: int,
    scaled_to_percent: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if scaled_to_percent:
        colorbar_label = f"{metric} (%)"
        if vmin is None:
            vmin = 50.0
        if vmax is None:
            vmax = 100.0
    else:
        colorbar_label = metric

    display_labels = [display_name(generator) for generator in generators]

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    masked_matrix = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(generators)), labels=display_labels)
    ax.set_yticks(np.arange(len(generators)), labels=display_labels)
    ax.set_xlabel("Testing Subset")
    ax.set_ylabel("Training Subset")
    ax.set_title(title or f"{metric} | train {train_compression} | test {test_compression}", weight="bold")

    ax.set_xticks(np.arange(-0.5, len(generators), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(generators), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_cells(ax, matrix, annot_format)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def print_sources(sources: list[list[EvalResult | None]], generators: list[str]) -> None:
    for row_idx, train_generator in enumerate(generators):
        for col_idx, eval_generator in enumerate(generators):
            result = sources[row_idx][col_idx]
            source = str(result.path) if result else "MISSING"
            print(f"{train_generator} -> {eval_generator}: {source}")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    train_compression = args.train_compression.lower()
    test_compression = args.test_compression.lower()
    preprocess_mode = args.preprocess_mode
    generators = [normalize_name(generator) for generator in args.generators]

    results = load_results(eval_dir)
    if not results:
        raise FileNotFoundError(f"No parseable *_metrics.json files found in {eval_dir}")

    matrix, sources = build_matrix(
        results=results,
        generators=generators,
        train_compression=train_compression,
        test_compression=test_compression,
        metric=args.metric,
        preprocess_mode=preprocess_mode,
    )

    scaled_to_percent = should_scale_to_percent(matrix, args.scale)
    if scaled_to_percent:
        matrix = matrix * 100.0

    output_path = Path(args.output) if args.output else default_output_path(args.metric, train_compression, test_compression)
    plot_heatmap(
        matrix=matrix,
        generators=generators,
        metric=args.metric,
        train_compression=train_compression,
        test_compression=test_compression,
        output_path=output_path,
        title=args.title,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        annot_format=args.annot_format,
        dpi=args.dpi,
        scaled_to_percent=scaled_to_percent,
    )

    if args.show_sources:
        print_sources(sources, generators)
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
