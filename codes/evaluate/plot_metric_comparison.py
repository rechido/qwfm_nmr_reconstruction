# ruff: noqa: E402
"""Plot comparable summaries produced by ``evaluate_saved_samples.py``.

This module never loads a model or a large ``samples.pt`` artifact. It reads
compact JSON summaries, validates the shared protocol and reference floor, and
draws model comparisons or numeric-NFE curves in the original plotting style.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.metrics import GROUND_METRICS
from codes.utils.io_utils import atomic_write_json


PLOT_PROTOCOL_VERSION = "continuous_posterior_metric_comparison_v2"
METRIC_ORDER = (
    "validity_distance",
    "coverage_distance",
    "energy_distance",
    "precision",
    "recall",
    "density",
    "coverage",
)
LOWER_IS_BETTER = frozenset(
    ("validity_distance", "coverage_distance", "energy_distance")
)
DISPLAY_NAMES = {
    "validity_distance": "Validity distance ↓",
    "coverage_distance": "Coverage distance ↓",
    "energy_distance": "Energy distance ↓",
    "precision": "Precision ↑",
    "recall": "Recall ↑",
    "density": "Density (interpret with coverage)",
    "coverage": "Coverage ↑",
}

# Keep the method colors used by the original multimode comparison galleries.
METHOD_COLORS = {
    "HFM": "#222222",
    "Histogram-Linear FM": "#222222",
    "QWFM-L128": "#1B9E77",
    "QWFM-L256": "#D95F02",
    "QWFM-L512": "#7570B3",
    "Regression-MSE": "#377EB8",
    "Regression-MSE-W1": "#56B4E9",
    "CNF": "#E41A1C",
    "CVAE-z16": "#E7298A",
    "CVAE-z32": "#AD1457",
    "NNLS": "#7F7F7F",
    "X2-D2": "#A6761D",
    "Discrepancy-D2": "#8C564B",
    "BayesReg-I-official": "#B15928",
    "BayesReg-I-known-noise": "#BCBD22",
}
_NFE_LABEL = re.compile(r"^(?P<method>.+)-NFE(?P<nfe>[0-9]+)$")


def _method_color(label: str) -> str:
    if label in METHOD_COLORS:
        return METHOD_COLORS[label]
    palette = ("#0072B2", "#CC79A7", "#009E73", "#E69F00", "#56B4E9")
    index = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")
    return palette[index % len(palette)]


def _nfe_series(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[int, str]], list[str], list[int]]:
    series: dict[str, dict[int, str]] = {}
    static: list[str] = []
    for label in payloads:
        match = _NFE_LABEL.fullmatch(label)
        if match is None:
            static.append(label)
            continue
        method, nfe = match["method"], int(match["nfe"])
        if nfe < 1:
            raise ValueError(f"NFE must be positive in label {label!r}.")
        points = series.setdefault(method, {})
        if nfe in points:
            raise ValueError(f"duplicate NFE={nfe} for method {method!r}.")
        points[nfe] = label
    if not series:
        raise ValueError(
            "an NFE sweep requires at least one LABEL-NFE<number> summary."
        )
    if set(series).intersection(static):
        raise ValueError("a method cannot be both NFE-dependent and NFE-independent.")
    nfes = sorted({nfe for points in series.values() for nfe in points})
    return series, static, nfes


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _parse_evaluations(
    values: list[str],
    *,
    skip_missing: bool = False,
    skipped: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    payloads: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"evaluation {value!r} must use LABEL=SUMMARY.json.")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in seen:
            raise ValueError("evaluation labels must be nonempty and unique.")
        seen.add(label)
        path = Path(raw_path).expanduser().resolve()
        if skip_missing and not path.exists():
            print(
                f"WARNING: skipping missing evaluation {label!r}: {path}",
                file=sys.stderr,
            )
            if skipped is not None:
                skipped.append({"label": label, "path": str(path)})
            continue
        paths[label] = path
        payloads[label] = _load_json(path)
    if not payloads:
        raise ValueError("at least one evaluation summary is required.")
    return paths, payloads


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing {key!r}.")
    return mapping[key]


def _finite_statistic(value: Any, *, context: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a statistic object.")
    result: dict[str, float] = {}
    for name in ("mean", "standard_deviation", "minimum", "maximum"):
        scalar = _required(value, name, context)
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise ValueError(f"{context}.{name} must be numeric.")
        scalar = float(scalar)
        if not math.isfinite(scalar):
            raise ValueError(f"{context}.{name} must be finite.")
        result[name] = scalar
    if result["standard_deviation"] < 0:
        raise ValueError(f"{context}.standard_deviation must be nonnegative.")
    return result


def validate_evaluation_summaries(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate common identities and return the comparison contract."""

    if not payloads:
        raise ValueError("at least one summary is required.")
    first_label = next(iter(payloads))
    first = payloads[first_label]
    identity_fields = (
        "protocol_version",
        "metric_protocol_version",
        "specification_hash",
        "primary_reference_run_id",
        "oracle_reference_run_id",
        "settings",
        "condition_indices",
    )
    identity = {
        field: _required(first, field, f"summary {first_label!r}")
        for field in identity_fields
    }
    for label, payload in payloads.items():
        for field, expected in identity.items():
            if _required(payload, field, f"summary {label!r}") != expected:
                raise ValueError(
                    f"summary {label!r} has a different {field}; "
                    "only like-for-like evaluations may be plotted together."
                )
        aggregate = _required(payload, "aggregate", f"summary {label!r}")
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"summary {label!r}.aggregate must be an object.")
        for role in ("model", "oracle"):
            role_payload = _required(aggregate, role, f"summary {label!r}.aggregate")
            ground = _required(
                role_payload,
                "ground_metrics",
                f"summary {label!r}.aggregate.{role}",
            )
            for distance in GROUND_METRICS:
                metrics = _required(
                    ground,
                    distance,
                    f"summary {label!r}.aggregate.{role}.ground_metrics",
                )
                for metric in METRIC_ORDER:
                    _finite_statistic(
                        _required(metrics, metric, f"{label}.{role}.{distance}"),
                        context=f"{label}.{role}.{distance}.{metric}",
                    )

    # The oracle calculation is independent of the candidate model.  A
    # mismatch therefore exposes different subsets or a corrupted summary.
    oracle_first = first["aggregate"]["oracle"]["ground_metrics"]
    for label, payload in payloads.items():
        oracle = payload["aggregate"]["oracle"]["ground_metrics"]
        for distance in GROUND_METRICS:
            for metric in METRIC_ORDER:
                for statistic in ("mean", "standard_deviation", "minimum", "maximum"):
                    value = float(oracle[distance][metric][statistic])
                    expected = float(oracle_first[distance][metric][statistic])
                    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
                        raise ValueError(
                            f"summary {label!r} has a different oracle result for "
                            f"{distance}.{metric}.{statistic}."
                        )
    return identity


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return stem or "comparison"


def _plot_metric_axis(
    axis: Any,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    ground_metric: str,
    metric: str,
    plot_kind: str,
    show_standard_deviation: bool,
) -> None:
    oracle = next(iter(payloads.values()))["aggregate"]["oracle"]["ground_metrics"]

    def statistic(label: str) -> Mapping[str, float]:
        return payloads[label]["aggregate"]["model"]["ground_metrics"][ground_metric][
            metric
        ]

    if plot_kind == "nfe_sweep":
        series, static, nfes = _nfe_series(payloads)
        for method, points in series.items():
            x_values = sorted(points)
            means = [float(statistic(points[nfe])["mean"]) for nfe in x_values]
            deviations = [
                float(statistic(points[nfe])["standard_deviation"]) for nfe in x_values
            ]
            color = _method_color(method)
            axis.plot(
                x_values,
                means,
                marker="o",
                linewidth=1.9,
                markersize=4.8,
                color=color,
                label=method,
            )
            if show_standard_deviation:
                axis.fill_between(
                    x_values,
                    [mean - sd for mean, sd in zip(means, deviations)],
                    [mean + sd for mean, sd in zip(means, deviations)],
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
        for label in static:
            value = statistic(label)
            mean = float(value["mean"])
            color = _method_color(label)
            axis.axhline(
                mean,
                color=color,
                linestyle="--",
                linewidth=1.6,
                label=f"{label} (NFE-independent)",
            )
            if show_standard_deviation:
                deviation = float(value["standard_deviation"])
                axis.axhspan(
                    mean - deviation, mean + deviation, color=color, alpha=0.06
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks(nfes, [str(nfe) for nfe in nfes])
        axis.minorticks_off()
        axis.set_xlabel("Euler NFE (flow-matching methods)")
    else:
        labels = list(payloads)
        for position, label in enumerate(labels):
            value = statistic(label)
            axis.errorbar(
                position,
                float(value["mean"]),
                yerr=float(value["standard_deviation"])
                if show_standard_deviation
                else None,
                fmt="o",
                color=_method_color(label),
                markersize=5,
                linewidth=1.2,
                capsize=2.5,
            )
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_xlabel("Method")

    oracle_mean = float(oracle[ground_metric][metric]["mean"])
    if show_standard_deviation:
        oracle_std = float(oracle[ground_metric][metric]["standard_deviation"])
        axis.axhspan(
            oracle_mean - oracle_std,
            oracle_mean + oracle_std,
            color="#64748b",
            alpha=0.10,
            linewidth=0,
        )
    axis.axhline(
        oracle_mean,
        color="#64748b",
        linestyle=":",
        linewidth=1.4,
        label="independent SMC oracle",
    )
    axis.set_title(DISPLAY_NAMES[metric], fontsize=11)
    axis.set_ylabel(DISPLAY_NAMES[metric])
    axis.grid(alpha=0.22)
    axis.tick_params(axis="both", labelsize=8)


def _plot_ground_metric(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    ground_metric: str,
    destination: Path,
    title: str | None,
    dpi: int,
    plot_kind: str = "comparison",
    show_standard_deviation: bool = True,
    metric_destinations: Mapping[str, Path] | None = None,
) -> None:
    labels = list(payloads)
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(
            max(18, 0.65 * len(labels) + 12) if plot_kind == "comparison" else 18,
            10,
        ),
    )
    for axis, metric in zip(axes.flat, METRIC_ORDER):
        _plot_metric_axis(
            axis,
            payloads,
            ground_metric=ground_metric,
            metric=metric,
            plot_kind=plot_kind,
            show_standard_deviation=show_standard_deviation,
        )
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(
        handles, legend_labels, loc="center", fontsize=9, frameon=False
    )
    heading = title or "Conditional posterior sample-set comparison"
    figure.suptitle(f"{heading} — {ground_metric.upper()} ground distance", fontsize=15)
    figure.text(
        0.5,
        0.012,
        (
            "Whiskers/bands: descriptive SD across condition/reference-subset records, not confidence intervals. "
            if show_standard_deviation
            else "Lines: mean across condition/reference-subset records. "
        )
        + "Gray dotted line: independent SMC oracle; density is not a standalone quality ranking.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.965))
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    for metric, metric_path in (metric_destinations or {}).items():
        detail_figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
        _plot_metric_axis(
            axis,
            payloads,
            ground_metric=ground_metric,
            metric=metric,
            plot_kind=plot_kind,
            show_standard_deviation=show_standard_deviation,
        )
        axis.set_title(
            f"{DISPLAY_NAMES[metric]} — {ground_metric.upper()}\n{heading}", fontsize=11
        )
        axis.legend(frameon=False, fontsize=8)
        detail_figure.savefig(
            metric_path, dpi=dpi, bbox_inches="tight", facecolor="white"
        )
        plt.close(detail_figure)


def plot_metric_comparison(
    payloads: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
    output_dir: str | Path,
    *,
    title: str | None = None,
    filename_prefix: str = "posterior_metrics",
    dpi: int = 180,
    overwrite: bool = False,
    plot_kind: str = "comparison",
    show_standard_deviation: bool | None = None,
    skipped_evaluations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    identity = validate_evaluation_summaries(payloads)
    if plot_kind not in ("comparison", "nfe_sweep"):
        raise ValueError("plot_kind must be 'comparison' or 'nfe_sweep'.")
    if plot_kind == "nfe_sweep":
        series, static, nfes = _nfe_series(payloads)
    else:
        series, static, nfes = {}, [], []
    if show_standard_deviation is None:
        show_standard_deviation = plot_kind == "comparison"
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi must be an integer of at least 72.")
    prefix = _safe_stem(filename_prefix)
    output = Path(output_dir).expanduser().resolve()
    destinations = {
        distance: output / f"{prefix}_{distance}.png" for distance in GROUND_METRICS
    }
    metric_destinations = (
        {
            distance: {
                metric: output / f"{prefix}_{distance}_{metric}.png"
                for metric in METRIC_ORDER
            }
            for distance in GROUND_METRICS
        }
        if plot_kind == "nfe_sweep"
        else {}
    )
    table_path = output / f"{prefix}.csv"
    manifest_path = output / f"{prefix}.json"
    all_paths = [
        *destinations.values(),
        table_path,
        manifest_path,
        *(path for group in metric_destinations.values() for path in group.values()),
    ]
    if not overwrite and any(path.exists() for path in all_paths):
        raise FileExistsError(
            f"comparison output already exists in {output}; pass --overwrite."
        )
    output.mkdir(parents=True, exist_ok=True)
    for distance, destination in destinations.items():
        _plot_ground_metric(
            payloads,
            ground_metric=distance,
            destination=destination,
            title=title,
            dpi=dpi,
            plot_kind=plot_kind,
            show_standard_deviation=show_standard_deviation,
            metric_destinations=metric_destinations.get(distance),
        )

    rows: list[dict[str, Any]] = []
    for label, payload in payloads.items():
        for distance in GROUND_METRICS:
            for metric in METRIC_ORDER:
                model = payload["aggregate"]["model"]["ground_metrics"][distance][
                    metric
                ]
                oracle = payload["aggregate"]["oracle"]["ground_metrics"][distance][
                    metric
                ]
                rows.append(
                    {
                        "method": label,
                        "ground_metric": distance,
                        "metric": metric,
                        "direction": (
                            "lower_is_better"
                            if metric in LOWER_IS_BETTER
                            else "interpret_with_coverage"
                            if metric == "density"
                            else "higher_is_better"
                        ),
                        "model_mean": float(model["mean"]),
                        "model_standard_deviation": float(model["standard_deviation"]),
                        "oracle_mean": float(oracle["mean"]),
                        "oracle_standard_deviation": float(
                            oracle["standard_deviation"]
                        ),
                        "model_minus_oracle": float(model["mean"])
                        - float(oracle["mean"]),
                    }
                )
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "plot_protocol_version": PLOT_PROTOCOL_VERSION,
        **identity,
        "title": title,
        "plot_kind": plot_kind,
        "show_standard_deviation": show_standard_deviation,
        "uncertainty_description": "descriptive SD, not a confidence interval",
        "method_order": list(payloads),
        "nfe_method_order": list(series),
        "nfe_values": nfes,
        "nfe_independent_methods": static,
        "source_summaries": {label: str(paths[label]) for label in payloads},
        "skipped_evaluations": list(skipped_evaluations or []),
        "figures": {name: str(path) for name, path in destinations.items()},
        "metric_figures": {
            distance: {metric: str(path) for metric, path in group.items()}
            for distance, group in metric_destinations.items()
        },
        "table": str(table_path),
        "metric_order": list(METRIC_ORDER),
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "rows": rows,
    }
    first_payload = next(iter(payloads.values()))
    if "truth_metrics" in first_payload:
        manifest["truth_metrics"] = first_payload["truth_metrics"]
        manifest["observation_metadata"] = first_payload.get("observation_metadata")
        manifest["comparison_interpretation"] = (
            "Agreement with an independent numerical posterior reference; no experimental spectrum ground-truth accuracy or calibration is measured."
        )
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot like-for-like cached posterior metric summaries."
    )
    parser.add_argument(
        "--evaluations",
        nargs="+",
        required=True,
        metavar="LABEL=EVALUATION_SUMMARY.json",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--filename_prefix", default="posterior_metrics")
    parser.add_argument(
        "--plot_kind", choices=("comparison", "nfe_sweep"), default="comparison"
    )
    parser.add_argument(
        "--show_standard_deviation",
        action="store_true",
        default=None,
        help="Show descriptive SD; default on for comparisons, off for NFE curves.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Explicitly skip missing files, with warnings and manifest records; malformed files still fail.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    skipped: list[dict[str, str]] = []
    paths, payloads = _parse_evaluations(
        args.evaluations, skip_missing=args.skip_missing, skipped=skipped
    )
    manifest = plot_metric_comparison(
        payloads,
        paths,
        args.output_dir,
        title=args.title,
        filename_prefix=args.filename_prefix,
        dpi=args.dpi,
        overwrite=args.overwrite,
        plot_kind=args.plot_kind,
        show_standard_deviation=args.show_standard_deviation,
        skipped_evaluations=skipped,
    )
    print(
        f"saved {len(manifest['figures']) + sum(len(group) for group in manifest['metric_figures'].values())} "
        f"metric figures and {manifest['table']}",
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "LOWER_IS_BETTER",
    "METRIC_ORDER",
    "PLOT_PROTOCOL_VERSION",
    "plot_metric_comparison",
    "validate_evaluation_summaries",
]
