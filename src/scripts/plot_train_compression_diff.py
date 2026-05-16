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
from matplotlib.colors import TwoSlopeNorm


EVAL_FILE_RE = re.compile(
    r"(?P<model>.+?)_train-(?P<train_generator>[^_]+)_comp-(?P<train_compression>[^_]+)"
    r"_eval-(?P<eval_generator>[^_]+)_test-(?P<test_compression>[^_]+)"
    r"_(?P<timestamp>\d{8}_\d{6})_metrics\.json$"
)

GENERATOR_ALIASES = {
    "adml": "adm",
}

DEFAULT_GENERATORS = ["biggan", "adm", "glide"]
METRIC_CHOICES = ("accuracy", "recall", "roc_auc", "loss")


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
        description=(
            "Generate a cross-generator heatmap showing the metric difference between "
            "two detector training compression levels."
        )
    )
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", choices=METRIC_CHOICES, default="accuracy")
    parser.add_argument("--baseline-train-compression", default="raw")
    parser.add_argument("--compare-train-compression", default="jpeg96")
    parser.add_argument("--test-compression", required=True)
    parser.add_argument(
        "--preprocess-mode",
        default=None,
        help="Optional filter for eval JSON payload preprocess_mode, e.g. standard.",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=DEFAULT_GENERATORS,
        help="Generator order for both axes. Defaults to: biggan adm glide.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--annot-format", default="+.2f")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any requested cell is missing instead of plotting N/A cells.",
    )
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


def display_metric(metric: str) -> str:
    mapping = {
        "accuracy": "Accuracy",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
        "loss": "Loss",
    }
    return mapping.get(metric, metric)


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


def metric_to_plot_units(value: float, metric: str) -> float:
    if metric in {"accuracy", "recall", "roc_auc"} and 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def build_matrix(
    results: list[EvalResult],
    generators: list[str],
    train_compression: str,
    test_compression: str,
    metric: str,
    preprocess_mode: str | None = None,
) -> tuple[np.ndarray, list[list[EvalResult | None]], list[str]]:
    matrix = np.full((len(generators), len(generators)), np.nan, dtype=float)
    sources: list[list[EvalResult | None]] = []
    missing: list[str] = []

    for row_idx, train_generator in enumerate(generators):
        source_row: list[EvalResult | None] = []
        for col_idx, eval_generator in enumerate(generators):
            result = pick_latest_result(
                results=results,
                train_generator=train_generator,
                eval_generator=eval_generator,
                train_compression=train_compression,
                test_compression=test_compression,
                preprocess_mode=preprocess_mode,
            )
            source_row.append(result)
            if result is None:
                missing.append(f"{train_compression}: {train_generator} -> {eval_generator}")
                continue

            value = result.metrics.get(metric)
            if value is None:
                missing.append(f"{train_compression}: {train_generator} -> {eval_generator} missing {metric}")
                continue
            matrix[row_idx, col_idx] = metric_to_plot_units(float(value), metric)
        sources.append(source_row)

    return matrix, sources, missing


def default_output_path(
    metric: str,
    baseline_train_compression: str,
    compare_train_compression: str,
    test_compression: str,
) -> Path:
    return (
        Path("outputs/plots")
        / (
            f"eval_heatmap_{metric}_diff_train-{compare_train_compression}"
            f"_minus-{baseline_train_compression}_test-{test_compression}.png"
        )
    )


def finite_or_default(values: np.ndarray, default: float) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return default
    return float(np.nanmax(np.abs(finite_values)))


def annotate_cells(ax: plt.Axes, matrix: np.ndarray, fmt: str, white_threshold: float) -> None:
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                text = format(value, fmt)
                color = "white" if abs(value) >= white_threshold else "#172033"
            else:
                text = "N/A"
                color = "#697386"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                weight="bold",
            )


def plot_diff_heatmap(
    diff_matrix: np.ndarray,
    generators: list[str],
    metric: str,
    baseline_train_compression: str,
    compare_train_compression: str,
    test_compression: str,
    output_path: Path,
    title: str | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    annot_format: str,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_abs_diff = finite_or_default(diff_matrix, 10.0)
    if vmin is None:
        vmin = -max_abs_diff
    if vmax is None:
        vmax = max_abs_diff

    display_labels = [display_name(generator) for generator in generators]
    metric_label = display_metric(metric)
    unit_label = "percentage points" if metric in {"accuracy", "recall", "roc_auc"} else metric

    fig, ax = plt.subplots(figsize=(7.2, 5.9), constrained_layout=True)
    masked_matrix = np.ma.masked_invalid(diff_matrix)
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(masked_matrix, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(generators)), labels=display_labels)
    ax.set_yticks(np.arange(len(generators)), labels=display_labels)
    ax.set_xlabel("Testing Subset", weight="bold")
    ax.set_ylabel("Training Subset", weight="bold")
    ax.set_title(
        title
        or (
            f"{metric_label}: train {compare_train_compression} - train {baseline_train_compression} "
            f"| test {test_compression}"
        ),
        weight="bold",
    )

    ax.set_xticks(np.arange(-0.5, len(generators), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(generators), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    white_threshold = max(abs(vmin), abs(vmax)) * 0.5
    annotate_cells(ax, diff_matrix, annot_format, white_threshold)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(unit_label)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def print_sources(
    label: str,
    sources: list[list[EvalResult | None]],
    generators: list[str],
) -> None:
    print(f"\n[{label}] selected sources")
    for row_idx, train_generator in enumerate(generators):
        for col_idx, eval_generator in enumerate(generators):
            result = sources[row_idx][col_idx]
            source = str(result.path) if result else "MISSING"
            print(f"{train_generator} -> {eval_generator}: {source}")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    generators = [normalize_name(generator) for generator in args.generators]
    baseline_train_compression = args.baseline_train_compression.lower()
    compare_train_compression = args.compare_train_compression.lower()
    test_compression = args.test_compression.lower()

    results = load_results(eval_dir)
    if not results:
        raise FileNotFoundError(f"No parseable *_metrics.json files found in {eval_dir}")

    baseline_matrix, baseline_sources, baseline_missing = build_matrix(
        results=results,
        generators=generators,
        train_compression=baseline_train_compression,
        test_compression=test_compression,
        metric=args.metric,
        preprocess_mode=args.preprocess_mode,
    )
    compare_matrix, compare_sources, compare_missing = build_matrix(
        results=results,
        generators=generators,
        train_compression=compare_train_compression,
        test_compression=test_compression,
        metric=args.metric,
        preprocess_mode=args.preprocess_mode,
    )
    missing = [*baseline_missing, *compare_missing]

    if args.show_sources:
        print_sources(baseline_train_compression, baseline_sources, generators)
        print_sources(compare_train_compression, compare_sources, generators)

    if missing:
        print("\nMissing requested cells:")
        for cell in missing:
            print(f"- {cell}")
    if args.strict_missing and missing:
        raise FileNotFoundError("Missing requested eval cells. See messages above.")

    diff_matrix = compare_matrix - baseline_matrix
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(
            metric=args.metric,
            baseline_train_compression=baseline_train_compression,
            compare_train_compression=compare_train_compression,
            test_compression=test_compression,
        )
    )
    plot_diff_heatmap(
        diff_matrix=diff_matrix,
        generators=generators,
        metric=args.metric,
        baseline_train_compression=baseline_train_compression,
        compare_train_compression=compare_train_compression,
        test_compression=test_compression,
        output_path=output_path,
        title=args.title,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        annot_format=args.annot_format,
        dpi=int(args.dpi),
    )
    print(f"Saved difference heatmap to {output_path}")


if __name__ == "__main__":
    main()
