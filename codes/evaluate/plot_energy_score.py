# ruff: noqa: E402
"""Plot truth-based energy and Gaussian kernel scores from saved reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.evaluate.plot_metric_comparison import METHOD_COLORS


def validate_comparison_reports(
    records: list[dict], *, score_name: str = "energy_score"
) -> dict:
    """Require the same stored simulator truth, sample count, and conditions."""
    if not records:
        raise ValueError("No available truth-score reports to compare.")
    first = records[0]["report"]
    first_truth_hash = first.get("source_sha256", {}).get("simulator_truth")
    for record in records:
        report = record["report"]
        truth_hash = report.get("source_sha256", {}).get("simulator_truth")
        if not isinstance(truth_hash, str) or not truth_hash.strip():
            raise ValueError("Truth-score comparisons require a nonempty simulator truth SHA256.")
        if truth_hash != first_truth_hash:
            raise ValueError("Truth-score reports use different simulator truth SHA256 values.")
        for key in ("specification_hash", "condition_count"):
            if report[key] != first[key]:
                raise ValueError(f"Cannot compare reports with different {key}.")
        # Independent evaluation stores a Dirac prediction only once. Its
        # requested ensemble budget is still shared with stochastic models.
        budget = report.get("requested_sample_count", report["sample_count"])
        first_budget = first.get("requested_sample_count", first["sample_count"])
        if budget != first_budget:
            raise ValueError("Cannot compare reports with different sample_count budgets.")
        if "requested_sample_count" in report:
            expected = 1 if report.get("deterministic") else budget
            if report["sample_count"] != expected:
                raise ValueError("Stored sample count differs from the declared prediction budget.")
        if [row["condition_id"] for row in report["per_condition"]] != [
            row["condition_id"] for row in first["per_condition"]
        ]:
            raise ValueError("Truth-score reports use different test conditions.")
        if score_name not in report.get("aggregate", {}):
            raise ValueError(f"Report does not contain {score_name}.")
        if score_name == "kernel_score":
            bandwidth = report.get("kernel_bandwidth")
            if (
                isinstance(bandwidth, bool) or not isinstance(bandwidth, (float, int))
                or not math.isfinite(bandwidth) or bandwidth <= 0
            ):
                raise ValueError("Kernel-score comparisons require a finite positive bandwidth.")
            for key in ("protocol_version", "kernel", "kernel_bandwidth"):
                if report.get(key) != first.get(key):
                    raise ValueError(f"Cannot compare kernel scores with different {key}.")
    return first


def _plot_score_axis(ax, records: list[dict], key: str, title: str) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method_label"]].append(record)
    for label, method_records in grouped.items():
        color = METHOD_COLORS.get(label)
        series = sorted(
            [record for record in method_records if record["nfe"] is not None],
            key=lambda record: record["nfe"],
        )
        if series:
            ax.errorbar(
                [record["nfe"] for record in series],
                [record["report"]["aggregate"][key]["mean"] for record in series],
                yerr=[
                    record["report"]["aggregate"][key]["standard_error"] or 0
                    for record in series
                ],
                marker="o",
                capsize=3,
                linewidth=1.7,
                label=label,
                color=color,
            )
        else:
            if len(method_records) != 1:
                raise ValueError(f"Ambiguous static method label: {label}")
            stats = method_records[0]["report"]["aggregate"][key]
            line = ax.axhline(
                stats["mean"],
                linestyle="--",
                linewidth=1.4,
                label=label,
                color=color,
            )
            standard_error = stats["standard_error"]
            if standard_error is not None:
                ax.axhspan(
                    stats["mean"] - standard_error,
                    stats["mean"] + standard_error,
                    color=line.get_color(),
                    alpha=0.08,
                    linewidth=0,
                    zorder=0,
                )
    ax.set_xscale("log", base=4)
    ax.set_xticks([1, 4, 16, 64, 256], labels=[1, 4, 16, 64, 256])
    ax.set_xlabel("Euler NFE (dashed: methods without Euler integration)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.22)


def _save_figure(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"saved: {path}")
    plt.close(fig)


def _plot_combined_scores(energy_records: list[dict], kernel_records: list[dict],
                          output: Path) -> None:
    first = validate_comparison_reports(energy_records, score_name="energy_score")
    kernel_first = validate_comparison_reports(kernel_records, score_name="kernel_score")
    # Each panel must describe the same predictions and test pairs, even when
    # the standalone command is given two independently saved summaries.
    def record_key(record):
        return record["method_label"], record["nfe"] if record["nfe"] is not None else -1

    energy_records = sorted(energy_records, key=record_key)
    kernel_records = sorted(kernel_records, key=record_key)
    if len(energy_records) != len(kernel_records) or any(
        record_key(energy) != record_key(kernel)
        or energy["report"]["source_sha256"] != kernel["report"]["source_sha256"]
        for energy, kernel in zip(energy_records, kernel_records)
    ):
        raise ValueError("Combined ES/KS figures require the same models, NFE settings, and prediction sources.")
    for key in ("specification_hash", "condition_count"):
        if first[key] != kernel_first[key]:
            raise ValueError(f"Combined ES/KS figures require the same {key}.")
    budget = first.get("requested_sample_count", first["sample_count"])
    if budget != kernel_first.get("requested_sample_count", kernel_first["sample_count"]) or (
        [row["condition_id"] for row in first["per_condition"]]
        != [row["condition_id"] for row in kernel_first["per_condition"]]
    ):
        raise ValueError("Combined ES/KS figures require the same sample budget and test conditions.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    _plot_score_axis(axes[0], energy_records, "energy_score", "Energy score (lower is better)")
    _plot_score_axis(axes[1], kernel_records, "kernel_score", "Kernel score (lower is better)")
    axes[0].set_ylabel("Histogram-vector L2 units")
    axes[1].set_ylabel("Dimensionless Gaussian kernel score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(
        f"Known simulator truth; G={first['condition_count']}, K={budget}; "
        f"Gaussian bandwidth={kernel_first['kernel_bandwidth']:.6g}\n"
        "Bars (flow) / bands (static): ±1 condition-level SE, not training-seed uncertainty",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.93))
    _save_figure(fig, output, "proper_scores_comparison")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--kernel_summary", type=Path,
                        help="With --score=both, the matching kernel-score summary (default: beside --summary).")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--score", choices=("energy_score", "kernel_score", "both"))
    args = parser.parse_args(argv)
    if args.kernel_summary is not None and args.score != "both":
        parser.error("--kernel_summary requires --score=both.")
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    score_name = args.score or (
        "kernel_score" if "kernel_score" in payload.get("protocol_version", "")
        else "energy_score"
    )
    records = [record for record in payload["records"] if record["available"]]
    if not records:
        print(f"No truth-based {score_name} values are available; no plot produced.")
        return
    if score_name == "both":
        kernel_path = args.kernel_summary or args.summary.with_name("kernel_score_summary.json")
        kernel_payload = json.loads(kernel_path.read_text(encoding="utf-8"))
        kernel_records = [record for record in kernel_payload["records"] if record["available"]]
        _plot_combined_scores(records, kernel_records, args.output_dir or args.summary.parent)
        return
    first = validate_comparison_reports(records, score_name=score_name)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    keys = ("energy_score", "mean_distance_to_truth", "mean_pairwise_distance")
    titles = (
        "Energy score = A - B/2 (lower is better)",
        "A: distance to truth (diagnostic)",
        "B: ensemble spread (diagnostic)",
    )
    if score_name == "kernel_score":
        keys = ("kernel_score", "mean_kernel_to_truth", "mean_pairwise_kernel")
        titles = (
            "Kernel score = D - 2C (lower is better)",
            "C: similarity to truth (diagnostic)",
            "D: within-ensemble similarity (diagnostic)",
        )
    for ax, key, title in zip(axes, keys, titles):
        _plot_score_axis(ax, records, key, title)
    axes[0].set_ylabel(
        "Histogram-vector L2 units" if score_name == "energy_score"
        else "Dimensionless Gaussian kernel score"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    kernel_label = (
        f"; Gaussian bandwidth={first['kernel_bandwidth']:.6g}"
        if score_name == "kernel_score" else ""
    )
    fig.suptitle(
        f"Known simulator truth; G={first['condition_count']}, K={first.get('requested_sample_count', first['sample_count'])}{kernel_label}\n"
        "Bars (flow) / bands (static): ±1 condition-level SE, not training-seed uncertainty",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    _save_figure(fig, args.output_dir or args.summary.parent, f"{score_name}_comparison")


if __name__ == "__main__":
    main()
