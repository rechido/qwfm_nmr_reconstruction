"""Visualize saved metrics for one model, one setting, or several settings.

Only the requested JSON/CSV evaluation reports are read. This command does not
load sample tensors, run a network, or recompute evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codes.evaluate.export_paper_results import (  # noqa: E402
    MODELS, SETTINGS, PaperProtocol, collect_reports, forward_table,
)


FLOW_MODELS = ("HFM", "QWFM")
COLORS = {"HFM": "#0072B2", "QWFM": "#D55E00", "Regression": "#009E73", "CVAE": "#CC79A7"}
LABELS = {
    "precision": "Precision (higher is better)",
    "recall": "Recall (higher is better)",
    "energy_score": "Energy score (lower is better)",
    "kernel_score": "Kernel score (lower is better)",
    "forward_individual": "Individual-spectrum error (lower is better)",
    "forward_mean_spectrum": "Mean-spectrum error (lower is better)",
}
PLOT_GROUPS = (
    ("precision_recall_l2", ("precision", "recall"), "l2", "Histogram L2 precision and recall"),
    ("precision_recall_w1", ("precision", "recall"), "w1", "Log-rate W1 precision and recall"),
    ("precision_recall_w2", ("precision", "recall"), "w2", "Log-rate W2 precision and recall"),
    ("proper_scores_l2", ("energy_score", "kernel_score"), "l2", "Histogram L2 proper scores"),
    ("proper_scores_w2", ("energy_score", "kernel_score"), "w2", "Exact log-rate W2 proper scores"),
    ("forward_consistency", ("forward_individual", "forward_mean_spectrum"),
     "signal_mse_over_sigma_squared", "Forward consistency: signal MSE / noise variance"),
)


def metrics_at_max_nfe(rows: list[dict]) -> list[dict]:
    """Summarize every setting at its largest selected flow budget.

Non-flow predictions retain an empty NFE. F1 is the harmonic mean of the
unrounded aggregate P/R, with no invented uncertainty estimate.
"""
    max_nfe: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["nfe"] is not None:
            key = row["setting"], row["model"]
            max_nfe[key] = max(max_nfe.get(key, 0), row["nfe"])
    selected = [row.copy() for row in rows if row["nfe"] is None or
                row["nfe"] == max_nfe[row["setting"], row["model"]]]
    index = {(row["setting"], row["model"], row["metric"], row["distance"]): row for row in selected}
    for row in tuple(selected):
        if row["metric"] != "precision":
            continue
        recall = index[row["setting"], row["model"], "recall", row["distance"]]
        total = row["mean"] + recall["mean"]
        selected.append({**row, "metric": "f1", "mean": 2 * row["mean"] * recall["mean"] / total if total else 0.0,
                         "standard_error": None})
    metric_order = {name: i for i, name in enumerate(
        ("precision", "recall", "f1", "energy_score", "kernel_score", "forward_individual", "forward_mean_spectrum"))}
    return sorted(selected, key=lambda row: (row["setting"], metric_order[row["metric"]],
                                             row["distance"], MODELS.index(row["model"])))


def _draw_panel(ax, rows: list[dict], models: tuple[str, ...], nfes: tuple[int, ...]) -> None:
    """Use NFE only for flow predictions; non-flow-only panels are categorical."""
    has_flow = any(model in FLOW_MODELS for model in models)
    for index, model in enumerate(models):
        values = sorted((row for row in rows if row["model"] == model), key=lambda row: row["nfe"] or 0)
        if not values:
            raise ValueError(f"No metric values available for selected model {model}.")
        if model in FLOW_MODELS:
            ax.errorbar([row["nfe"] for row in values], [row["mean"] for row in values],
                        yerr=[row["standard_error"] for row in values], color=COLORS[model],
                        marker="o", markersize=4, linewidth=1.6, capsize=3, label=model)
        elif has_flow:
            value, = values
            ax.axhline(value["mean"], color=COLORS[model], linestyle="--" if model == "Regression" else "-.",
                       linewidth=1.6, label=f"{model} (NFE independent)")
            ax.axhspan(value["mean"] - value["standard_error"], value["mean"] + value["standard_error"],
                       color=COLORS[model], alpha=.12)
        else:
            value, = values
            ax.errorbar(index, value["mean"], yerr=value["standard_error"], color=COLORS[model],
                        marker="o", markersize=6, capsize=4, linestyle="none", label=model)
            ax.annotate(f"{value['mean']:.4g}", (index, value["mean"]), xytext=(8, 6),
                        textcoords="offset points", fontsize=8, color=COLORS[model])
    if has_flow:
        ax.set_xscale("log", base=4)
        ax.set_xticks(nfes, labels=[str(nfe) for nfe in nfes])
        ax.set_xlabel("NFE")
        if len(nfes) == 1:
            ax.set_xlim(nfes[0] / 1.8, nfes[0] * 1.8)
    else:
        ax.set_xticks(range(len(models)), labels=models)
        ax.set_xlim(-.6, len(models) - .4)
        ax.set_xlabel("Model (NFE independent)")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def plot_metrics(rows: list[dict], protocol: PaperProtocol, models: tuple[str, ...], output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    for stem, metrics, distance, title in PLOT_GROUPS:
        figure, axes = plt.subplots(len(metrics), len(protocol.settings), squeeze=False, sharey="row",
                                    figsize=(5.1 * len(protocol.settings), 6.6), constrained_layout=True)
        for column, (m, n) in enumerate(protocol.settings):
            for row_index, metric in enumerate(metrics):
                ax = axes[row_index, column]
                selected = [row for row in rows if row["setting"] == f"m{m}n{n}" and
                            row["metric"] == metric and row["distance"] == distance]
                _draw_panel(ax, selected, models, protocol.nfes)
                if column == 0:
                    ax.set_ylabel(LABELS[metric], fontsize=9)
                if row_index == 0:
                    ax.set_title(f"M={m}, N={n}")
        axes[0, 0].legend(fontsize=8, loc="best")
        figure.suptitle(f"{title}\nMean ± 1 standard error across test conditions", fontsize=11)
        for extension in ("png", "pdf"):
            destination = output_dir / f"{stem}.{extension}"
            figure.savefig(destination, dpi=220, facecolor="white")
            outputs.append(destination)
        plt.close(figure)
    return outputs


def export_results(experiments_root: Path, output_dir: Path, protocol: PaperProtocol = PaperProtocol(), *,
                   models: tuple[str, ...] = MODELS, plots: bool = True) -> dict:
    """Export one completed model or any explicit comparison of completed runs."""
    if not protocol.settings or len(set(protocol.settings)) != len(protocol.settings):
        raise ValueError("Select at least one unique experiment setting.")
    if not protocol.nfes or any(nfe < 1 for nfe in protocol.nfes) or tuple(sorted(set(protocol.nfes))) != protocol.nfes:
        raise ValueError("NFE budgets must be unique positive integers in increasing order.")
    # All selected reports are validated before creating any output files.
    rows, manifest = collect_reports(experiments_root, protocol, models=models)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"all_metrics.csv": rows, "metrics_at_max_nfe.csv": metrics_at_max_nfe(rows),
              "forward_consistency.csv": forward_table(rows)}
    outputs: list[Path] = []
    for name, records in tables.items():
        destination = output_dir / name
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        outputs.append(destination)
    if plots:
        outputs.extend(plot_metrics(rows, protocol, models, output_dir))
    level = "multiple_settings" if len(protocol.settings) > 1 else "single_model" if len(models) == 1 else "setting_comparison"
    manifest.update(protocol_version="selected_results_visualization_v1", visualization_level=level,
                    metric_row_count=len(rows), plots_generated=plots,
                    plot_error_bars="one standard error across test conditions",
                    summary_f1="harmonic mean of unrounded aggregate precision and recall; no F1 uncertainty estimated",
                    nonflow_nfe="not applicable; categorical points alone, horizontal references in flow comparisons",
                    outputs_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiments_root", type=Path, default=Path("outputs"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", choices=[f"m{m}n{n}" for m, n in SETTINGS], required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--nfes", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--skip_plots", action="store_true", help="Write CSV tables and provenance only.")
    args = parser.parse_args(argv)
    for name in ("settings", "models", "nfes"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name} must not contain duplicates.")
    if any(nfe < 1 for nfe in args.nfes):
        parser.error("--nfes must contain positive integers.")
    args.nfes = tuple(sorted(args.nfes))
    args.settings = tuple(pair for pair in SETTINGS if f"m{pair[0]}n{pair[1]}" in args.settings)
    args.models = tuple(model for model in MODELS if model in args.models)
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    try:
        manifest = export_results(args.experiments_root.expanduser().resolve(), args.output_dir.expanduser().resolve(),
                                  PaperProtocol(settings=args.settings, nfes=args.nfes), models=args.models,
                                  plots=not args.skip_plots)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Evaluation report schema is incomplete or incompatible: {error}") from error
    print(f"Saved {manifest['visualization_level']} visualizations and {manifest['metric_row_count']} metric rows to {args.output_dir}.")
    return manifest


if __name__ == "__main__":
    main()
