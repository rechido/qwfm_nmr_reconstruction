# ruff: noqa: E402
"""Compute truth-based energy and Gaussian kernel scores without inference.

Only simulator truths are read from a fixed split/reference container. SMC
posterior samples are never used as truth. Existing metric trees can optionally
be extended with two top-level fields; their previous contents are backed up.
The historical entrypoint and energy-score paths remain supported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.metrics.energy_score import compute_energy_score_report
from codes.metrics.kernel_score import compute_kernel_score_report
from codes.utils.io_utils import atomic_write_json, load_torch_artifact


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_truth_file(sample_path: Path, artifact: dict[str, Any]) -> Path | None:
    """Find a matching small, original simulator split before any SMC bank."""
    for parent in sample_path.parents:
        candidates = sorted(parent.glob("reference*/test_conditions.pt"))
        if (parent / "test_conditions.pt").is_file():
            candidates.append(parent / "test_conditions.pt")
        matching = []
        for path in candidates:
            truth = load_torch_artifact(path)
            if truth.get("protocol_id") == artifact.get("protocol_id") and truth.get(
                "specification_hash"
            ) == artifact.get("specification_hash"):
                matching.append((path, truth))
        if matching:
            first_path, first = matching[0]
            for other_path, other in matching[1:]:
                for field in ("sample_id", "measurement", "histogram"):
                    if not torch.equal(first[field], other[field]):
                        raise ValueError(
                            f"Conflicting simulator truths: {first_path}, {other_path}; "
                            "select --truth_file explicitly."
                        )
            return first_path
        if parent == PROJECT_ROOT or parent == parent.parent:
            break
    return None


def _method_label(artifact: dict[str, Any], sample_path: Path) -> str:
    method = artifact.get("method", "unknown")
    config = artifact.get("training_config", {})
    if method == "quantile_wasserstein":
        return f"QWFM-L{config.get('num_quantiles', artifact.get('state_size', '?'))}"
    if method == "histogram_linear":
        return "HFM"
    if method == "conditional_vae":
        return f"CVAE-z{config.get('latent_dim', config.get('latent_size', '?'))}"
    if method == "conditional_normalizing_flow":
        return "CNF"
    if str(method).startswith("deterministic_regression"):
        loss = str(config.get("regression_loss", ""))
        return (
            "Regression-MSE-W1"
            if "w1" in loss or "mse_w1" in str(sample_path)
            else "Regression-MSE"
        )
    return str(method)


def _merge_existing_metrics(
    metrics_dir: Path, report: dict[str, Any], kernel_report: dict[str, Any]
) -> list[str]:
    # Validate both files before changing either one. The spectrum specification
    # alone does not identify the evaluated conditions or predictive ensemble.
    prepared = []
    for filename in ("evaluation_raw.json", "evaluation_summary.json"):
        path = metrics_dir / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("specification_hash") != report.get("specification_hash"):
            raise ValueError(f"Existing metrics have a different specification: {path}")
        previous_count = payload.get("settings", {}).get("sample_count")
        if previous_count is not None and previous_count != report["sample_count"]:
            raise ValueError(f"Existing metrics use a different sample count: {path}")
        previous_indices = payload.get("condition_indices")
        if (
            previous_indices is not None
            and previous_indices != report["condition_indices"]
        ):
            raise ValueError(f"Existing metrics use different test conditions: {path}")
        for provenance in (
            payload, payload.get("energy_score", {}), payload.get("kernel_score", {})
        ):
            old_hash = provenance.get("source_sha256", {}).get("samples")
            if old_hash is not None:
                if old_hash != report["source_sha256"]["samples"]:
                    raise ValueError(
                        f"Existing metrics identify different samples: {path}"
                    )
                continue
            old_path = provenance.get("source_paths", {}).get("samples")
            if old_path is not None:
                recorded_path = Path(old_path)
                if not recorded_path.is_absolute():
                    recorded_path = PROJECT_ROOT / recorded_path
                if (
                    recorded_path.resolve()
                    != Path(report["source_paths"]["samples"]).resolve()
                ):
                    if (
                        not recorded_path.is_file()
                        or file_sha256(recorded_path)
                        != report["source_sha256"]["samples"]
                    ):
                        raise ValueError(
                            f"Cannot match the recorded sample source: {path}"
                        )
        prepared.append((path, payload))
    updated = []
    for path, payload in prepared:
        backup = path.with_name(path.stem + ".before_energy_score.json")
        if not backup.exists():
            shutil.copy2(path, backup)
        kernel_backup = path.with_name(path.stem + ".before_kernel_score.json")
        if not kernel_backup.exists():
            shutil.copy2(path, kernel_backup)
        payload["energy_score"] = report
        payload["kernel_score"] = kernel_report
        atomic_write_json(path, payload)
        updated.append(str(path))
    return updated


def evaluate_one(
    sample_path: Path,
    *,
    truth_file: Path | None = None,
    output_dir: Path | None = None,
    sample_count: int | None = None,
    distance_chunk_size: int = 64,
    kernel_bandwidth: float | None = None,
    update_existing_metrics: bool = False,
) -> dict[str, Any]:
    artifact = load_torch_artifact(sample_path)
    metrics_dir = sample_path.parent / "metrics"
    selected = None
    summary_path = metrics_dir / "evaluation_summary.json"
    if update_existing_metrics and summary_path.is_file():
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = old_summary.get("condition_indices")
        previous_count = old_summary.get("settings", {}).get("sample_count")
        if sample_count is not None and previous_count != sample_count:
            raise ValueError("--sample_count differs from the existing evaluation.")
        if sample_count is None:
            sample_count = previous_count
    source_truth = truth_file
    if source_truth is None:
        source_truth = find_truth_file(sample_path, artifact)
    truth = load_torch_artifact(source_truth) if source_truth else None
    report = compute_energy_score_report(
        artifact,
        truth,
        sample_count=sample_count,
        condition_indices=selected,
        distance_chunk_size=distance_chunk_size,
        device="cpu",
    )
    kernel_report = compute_kernel_score_report(
        artifact,
        truth,
        sample_count=sample_count,
        condition_indices=selected,
        distance_chunk_size=distance_chunk_size,
        device="cpu",
        bandwidth=kernel_bandwidth,
    )
    report["specification_hash"] = artifact["specification_hash"]
    report["source_paths"] = {
        "samples": str(sample_path),
        "simulator_truth": str(source_truth) if source_truth else None,
    }
    report["source_sha256"] = {
        "samples": file_sha256(sample_path),
        "simulator_truth": file_sha256(source_truth) if source_truth else None,
    }
    report["network_forward_performed"] = False
    report["smc_posterior_samples_used"] = False
    report["method_label"] = _method_label(artifact, sample_path)
    report["nfe"] = artifact.get("sampling_config", {}).get("ode_steps")
    for key in (
        "source_paths", "source_sha256", "network_forward_performed",
        "smc_posterior_samples_used", "method_label", "nfe",
    ):
        kernel_report[key] = report[key]
    destination = (output_dir or metrics_dir) / "energy_score.json"
    kernel_destination = (output_dir or metrics_dir) / "kernel_score.json"
    updated = (
        _merge_existing_metrics(metrics_dir, report, kernel_report)
        if update_existing_metrics else []
    )
    atomic_write_json(destination, report)
    atomic_write_json(kernel_destination, kernel_report)
    return {
        "samples": str(sample_path),
        "report_path": str(destination),
        "available": report["available"],
        "status": report["status"],
        "method_label": report["method_label"],
        "nfe": report["nfe"],
        "report": report,
        "kernel_report_path": str(kernel_destination),
        "kernel_report": kernel_report,
        "updated_metrics": updated,
    }


def _write_csv(
    path: Path, records: list[dict[str, Any]], *, score_name: str = "energy_score"
) -> None:
    terms = (
        ("mean_distance_to_truth", "mean_pairwise_distance")
        if score_name == "energy_score"
        else ("mean_kernel_to_truth", "mean_pairwise_kernel")
    )
    fields = [
        "method",
        "nfe",
        "available",
        "ground_metric",
        "condition_count",
        "sample_count",
        f"{score_name}_mean",
        f"{score_name}_standard_error",
        *terms,
        *(["kernel_bandwidth", "bandwidth_selection"] if score_name == "kernel_score" else []),
        "report_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            report = record["report"]
            aggregate = report.get("aggregate", {})
            score = aggregate.get(score_name, {})
            writer.writerow(
                {
                    "method": record["method_label"],
                    "nfe": record["nfe"],
                    "available": record["available"],
                    "ground_metric": report.get("ground_metric", report.get("metric", "l2")),
                    "condition_count": report.get("condition_count"),
                    "sample_count": report.get("sample_count"),
                    f"{score_name}_mean": score.get("mean"),
                    f"{score_name}_standard_error": score.get("standard_error"),
                    **{term: aggregate.get(term, {}).get("mean") for term in terms},
                    **({
                        "kernel_bandwidth": report.get("kernel_bandwidth"),
                        "bandwidth_selection": report.get("bandwidth_provenance", {}).get("selection"),
                    } if score_name == "kernel_score" else {}),
                    "report_path": record["report_path"],
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--samples", type=Path)
    group.add_argument("--experiment_dir", type=Path)
    parser.add_argument("--truth_file", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--sample_count", type=int)
    parser.add_argument("--distance_chunk_size", type=int, default=64)
    parser.add_argument("--kernel_bandwidth", type=float, default=None,
                        help="Fixed Gaussian bandwidth; default: independent simulator calibration median.")
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    parser.add_argument("--update_existing_metrics", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_cpu_threads < 1:
        raise ValueError("--num_cpu_threads must be positive.")
    if args.kernel_bandwidth is not None and (
        not math.isfinite(args.kernel_bandwidth) or args.kernel_bandwidth <= 0
    ):
        raise ValueError("--kernel_bandwidth must be finite and positive.")
    torch.set_num_threads(args.num_cpu_threads)
    truth_path = args.truth_file.resolve() if args.truth_file else None
    if args.samples:
        result = evaluate_one(
            args.samples.resolve(),
            truth_file=truth_path,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            sample_count=args.sample_count,
            distance_chunk_size=args.distance_chunk_size,
            kernel_bandwidth=args.kernel_bandwidth,
            update_existing_metrics=args.update_existing_metrics,
        )
        print(f"{result['status']}: {result['report_path']}")
        print(f"{result['kernel_report']['status']}: {result['kernel_report_path']}")
        return
    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(experiment_dir)
    # This legacy backfill targets the original posterior-evaluation runs.
    # Independent proper-score and diagnostic banks have their own truths and
    # must not be mixed into these historical summaries.
    sample_paths = sorted(experiment_dir.glob("*/evaluation/**/samples.pt"))
    destination = (
        args.output_dir or experiment_dir / "metric_comparisons" / "energy_score"
    ).resolve()
    kernel_destination = (
        args.output_dir or experiment_dir / "metric_comparisons" / "kernel_score"
    ).resolve()
    records = []
    kernel_records = []
    for index, sample_path in enumerate(sample_paths):
        result = evaluate_one(
            sample_path,
            truth_file=truth_path,
            output_dir=(
                destination / "models" / sample_path.parent.relative_to(experiment_dir)
                if args.output_dir else None
            ),
            sample_count=args.sample_count,
            distance_chunk_size=args.distance_chunk_size,
            kernel_bandwidth=args.kernel_bandwidth,
            update_existing_metrics=args.update_existing_metrics,
        )
        kernel_report = result.pop("kernel_report")
        kernel_report_path = result.pop("kernel_report_path")
        records.append(result)
        kernel_records.append({
            **result,
            "available": kernel_report["available"],
            "status": kernel_report["status"],
            "report_path": kernel_report_path,
            "report": kernel_report,
        })
        print(
            f"[{index + 1}/{len(sample_paths)}] {result['status']}: {sample_path}",
            flush=True,
        )
    summary = {
        "protocol_version": "saved_truth_energy_score_batch_v1",
        "experiment_dir": str(experiment_dir),
        "status": "completed" if sample_paths else "no_saved_samples",
        "network_forward_performed": False,
        "sample_artifact_count": len(records),
        "computed_count": sum(record["available"] for record in records),
        "unavailable_count": sum(not record["available"] for record in records),
        "records": records,
        "missing_sample_policy": "Never load checkpoints or generate model samples.",
    }
    atomic_write_json(destination / "energy_score_summary.json", summary)
    _write_csv(destination / "energy_score_summary.csv", records)
    print(f"saved: {destination / 'energy_score_summary.json'}")
    kernel_summary = {
        **summary,
        "protocol_version": "saved_truth_kernel_score_batch_v1",
        "computed_count": sum(record["available"] for record in kernel_records),
        "unavailable_count": sum(not record["available"] for record in kernel_records),
        "records": kernel_records,
    }
    atomic_write_json(kernel_destination / "kernel_score_summary.json", kernel_summary)
    _write_csv(kernel_destination / "kernel_score_summary.csv", kernel_records,
               score_name="kernel_score")
    print(f"saved: {kernel_destination / 'kernel_score_summary.json'}")


if __name__ == "__main__":
    main()
