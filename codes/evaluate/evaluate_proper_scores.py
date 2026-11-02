"""Evaluate independent joint test pairs with ES/KS only, without inference.

Each condition contributes one score against its simulator truth. There are no
posterior references, PRDC calculations, or G/K resampling repetitions here.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from codes.data.spec import PROTOCOL_ID  # noqa: E402
from codes.evaluate.evaluate_energy_score import _method_label, _write_csv  # noqa: E402
from codes.evaluate.plot_energy_score import main as plot_scores, validate_comparison_reports  # noqa: E402
from codes.metrics.energy_score import compute_energy_score_report  # noqa: E402
from codes.metrics.kernel_score import compute_kernel_score_report  # noqa: E402
from codes.utils.io_utils import atomic_write_json  # noqa: E402
from codes.utils.proper_score_locks import file_lock  # noqa: E402
from codes.utils.predictive_generation_utils import (  # noqa: E402
    PROPER_SCORE_PROTOCOL_VERSION, PROPER_SCORE_TEST_SPLIT, _verified_cache, file_sha256,
)


CLASSICAL_LABELS = {
    "classic_nnls": "NNLS",
    "classic_x2_d2": "X2-D2",
    "classic_discrepancy_d2": "Discrepancy-D2",
    "classic_bayesreg_i_official": "BayesReg-I-official",
    "classic_bayesreg_i_known_noise": "BayesReg-I-known-noise",
}


def _validate_truth(truth: dict, groups: int) -> None:
    provenance = truth.get("proper_score_provenance", {})
    if (
        provenance.get("protocol_version") != PROPER_SCORE_PROTOCOL_VERSION
        or provenance.get("split") != PROPER_SCORE_TEST_SPLIT
        or truth.get("split") != PROPER_SCORE_TEST_SPLIT
        or provenance.get("num_conditions") != groups
        or len(truth.get("sample_id", [])) != groups
        or provenance.get("master_seed") != truth.get("master_seed")
        or not isinstance(provenance.get("master_seed"), int)
        or provenance.get("specification_hash") != truth.get("specification_hash")
    ):
        raise ValueError("Expected the independent proper-score test split with the requested G.")
    if truth.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Independent proper-score evaluation requires the synthetic protocol.")
    if not torch.equal(truth["sample_id"], torch.arange(groups, dtype=truth["sample_id"].dtype)):
        raise ValueError("Independent proper-score conditions require ordered, contiguous IDs.")


def _validate_samples(artifact: dict, truth_hash: str, groups: int, budget: int) -> dict:
    provenance = artifact.get("proper_score_provenance", {})
    if (
        provenance.get("protocol_version") != PROPER_SCORE_PROTOCOL_VERSION
        or provenance.get("conditions_sha256") != truth_hash
        or provenance.get("requested_sample_count") != budget
        or not provenance.get("variant")
    ):
        raise ValueError("Samples do not belong to this independent test split / K budget.")
    samples = artifact.get("samples")
    deterministic = artifact.get("deterministic")
    if not isinstance(deterministic, bool):
        raise ValueError("Independent score samples must explicitly declare deterministic forecasts.")
    used = 1 if deterministic else budget
    if (
        not isinstance(samples, torch.Tensor) or samples.ndim != 3
        or samples.shape[:2] != (groups, used)
        or provenance.get("stored_sample_count") != used
    ):
        raise ValueError("Samples must store exactly G x K predictions (G x 1 for point forecasts).")
    return provenance


def _score(path: Path, artifact: dict, truth: dict, truth_path: Path, truth_hash: str,
           output: Path, args: argparse.Namespace, publication_lock: Path) -> None:
    provenance = _validate_samples(artifact, truth_hash, args.num_conditions, args.sample_count)
    used = artifact["samples"].shape[1]
    common = {
        "evaluation_protocol": PROPER_SCORE_PROTOCOL_VERSION,
        "requested_sample_count": args.sample_count,
        "source_paths": {"samples": str(path), "simulator_truth": str(truth_path)},
        "source_sha256": {"samples": file_sha256(path), "simulator_truth": truth_hash},
        "network_forward_performed": False,
        "smc_posterior_samples_used": False,
        "variant": provenance["variant"],
        "method_label": CLASSICAL_LABELS.get(provenance["variant"], _method_label(artifact, path)),
        "nfe": artifact.get("sampling_config", {}).get("ode_steps"),
        "generation_provenance": provenance,
        "kernel_bandwidth_override": args.kernel_bandwidth,
    }
    reports = {}
    for name in ("energy_score", "kernel_score"):
        saved = output / f"{name}.json"
        if saved.exists():
            report = json.loads(saved.read_text())
            if any(report.get(key) != value for key, value in common.items()):
                raise ValueError(f"Existing report uses a different source/configuration: {saved}; use a new output directory.")
            if report.get("sample_count") != used or report.get("condition_count") != args.num_conditions:
                raise ValueError(f"Existing report has a different G/K: {saved}.")
            reports[name] = report
    if len(reports) == 2:
        print(f"cached scores: {path}", flush=True)
        return
    # Existing metric implementations validate IDs, observations, grid/protocol
    # identity and deterministic declarations before scoring each condition.
    energy = compute_energy_score_report(
        artifact, truth, sample_count=used, distance_chunk_size=args.distance_chunk_size,
    )
    kernel = compute_kernel_score_report(
        artifact, truth, sample_count=used, distance_chunk_size=args.distance_chunk_size,
        bandwidth=args.kernel_bandwidth,
    )
    with file_lock(publication_lock, blocking=True):
        for name, report in (("energy_score", energy), ("kernel_score", kernel)):
            if not report["available"]:
                raise ValueError(report.get("reason", f"{name} unavailable"))
            report.update(common)
            atomic_write_json(output / f"{name}.json", report)
    print(f"scored: {path}", flush=True)


def _collect_reports(output: Path, truth_hash: str, args: argparse.Namespace) -> dict[str, list[dict]]:
    records = {"energy_score": [], "kernel_score": []}
    for name in records:
        for path in sorted((output / "models").rglob(f"{name}.json")):
            report = json.loads(path.read_text())
            if (
                report.get("evaluation_protocol") != PROPER_SCORE_PROTOCOL_VERSION
                or report.get("source_sha256", {}).get("simulator_truth") != truth_hash
                or report.get("condition_count") != args.num_conditions
                or report.get("requested_sample_count") != args.sample_count
                or report.get("kernel_bandwidth_override") != args.kernel_bandwidth
            ):
                raise ValueError(f"Incompatible reports must not be pooled: {path}.")
            sample_path = Path(report["source_paths"]["samples"])
            if not sample_path.is_file() or file_sha256(sample_path) != report["source_sha256"]["samples"]:
                raise ValueError(f"Prediction source changed after scoring: {sample_path}.")
            records[name].append({
                "samples": str(sample_path), "report_path": str(path),
                "available": report["available"], "status": report["status"],
                "method_label": report["method_label"], "nfe": report["nfe"],
                "report": report,
            })
        validate_comparison_reports(records[name], score_name=name)
    if [r["samples"] for r in records["energy_score"]] != [r["samples"] for r in records["kernel_score"]]:
        raise ValueError("Energy and kernel reports must cover the same prediction artifacts.")
    return records


def _paired_records(records: list[dict], metric: str) -> list[dict]:
    result = []
    for left, right in itertools.combinations(records, 2):
        values = torch.tensor([
            a[metric] - b[metric]
            for a, b in zip(left["report"]["per_condition"], right["report"]["per_condition"])
        ], dtype=torch.float64)
        sd = float(values.std(correction=1)) if len(values) > 1 else None
        result.append({
            "metric": metric, "left_method": left["method_label"], "left_nfe": left["nfe"],
            "right_method": right["method_label"], "right_nfe": right["nfe"],
            "mean_difference_left_minus_right": float(values.mean()),
            "standard_deviation": sd,
            "standard_error": sd / math.sqrt(len(values)) if sd is not None else None,
            "condition_count": len(values),
        })
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--variant", default="all", help="One generated model variant, or all available models.")
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--distance_chunk_size", type=int, default=64)
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    parser.add_argument("--kernel_bandwidth", type=float)
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args(argv)
    for key in ("num_conditions", "sample_count", "distance_chunk_size", "num_cpu_threads"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive.")
    if args.sample_count < 2:
        parser.error("Independent stochastic score evaluation requires K>=2.")
    if args.kernel_bandwidth is not None and (not math.isfinite(args.kernel_bandwidth) or args.kernel_bandwidth <= 0):
        parser.error("kernel_bandwidth must be finite and positive.")
    torch.set_num_threads(args.num_cpu_threads)
    experiment = args.experiment_dir.expanduser().resolve()
    name = f"g{args.num_conditions}_k{args.sample_count}"
    run = (args.run_dir or experiment / "proper_scores" / name).expanduser().resolve()
    output = (args.output_dir or experiment / "metric_comparisons" / "proper_scores" / name).expanduser().resolve()
    truth_path = run / "test_conditions.pt"
    truth = _verified_cache(truth_path, True)
    if truth is None:
        raise FileNotFoundError(f"Prepare test conditions first with proper_scores/00_prepare_test_conditions.sh: {truth_path}")
    _validate_truth(truth, args.num_conditions)
    truth_hash = file_sha256(truth_path)
    selected = []
    # Prevalidate all selected sources before updating their reports.
    for path in sorted(run.rglob("samples.pt")):
        if not path.with_suffix(".sha256.json").is_file():
            print(f"not published yet (checksum sidecar missing), skipped: {path}", flush=True)
            continue
        artifact = _verified_cache(path, True)
        provenance = _validate_samples(artifact, truth_hash, args.num_conditions, args.sample_count)
        if args.variant == "all" or provenance["variant"] == args.variant:
            selected.append(path)
    if not selected:
        raise ValueError(f"No generated samples for variant={args.variant!r}; no inference is performed by this command.")
    for index, path in enumerate(selected):
        print(f"[{index + 1}/{len(selected)}] ES/KS only", flush=True)
        artifact = _verified_cache(path, True)
        model_output = output / "models" / path.parent.relative_to(run)
        with file_lock(model_output / ".score.lock", blocking=True):
            _score(path, artifact, truth, truth_path, truth_hash,
                   model_output, args, output / ".publication.lock")
    # Different models compute independently. Only paired report publication
    # and comparison-file updates are serialized, preventing lost summaries.
    with file_lock(output / ".publication.lock", blocking=True):
        collected = _collect_reports(output, truth_hash, args)
        paired = []
        summaries = {}
        for metric, records in collected.items():
            summary = {
                "protocol_version": f"independent_{metric}_batch_v1",
                "evaluation_protocol": PROPER_SCORE_PROTOCOL_VERSION,
                "experiment_dir": str(experiment), "run_dir": str(run),
                "condition_count": args.num_conditions, "requested_sample_count": args.sample_count,
                "network_forward_performed": False, "smc_posterior_samples_used": False,
                "source_sha256": {"simulator_truth": truth_hash},
                "sample_artifact_count": len(records), "computed_count": len(records),
                "unavailable_count": 0, "status": "completed", "records": records,
                "comparison_scope": "All compatible reports already evaluated in this output directory; missing models are not inferred.",
            }
            destination = output / f"{metric}_summary.json"
            atomic_write_json(destination, summary)
            _write_csv(output / f"{metric}_summary.csv", records, score_name=metric)
            paired.extend(_paired_records(records, metric))
            summaries[metric] = summary
            if not args.skip_plots:
                plot_scores(["--summary", str(destination), "--score", metric,
                             "--output_dir", str(output / "figures" / metric)])
            print(f"saved: {destination}", flush=True)
        if not args.skip_plots:
            plot_scores(["--summary", str(output / "energy_score_summary.json"),
                         "--kernel_summary", str(output / "kernel_score_summary.json"),
                         "--score", "both", "--output_dir", str(output / "figures")])
        atomic_write_json(output / "paired_score_differences.json", {
            "difference": "left_minus_right; lower score is better",
            "uncertainty": "SD/sqrt(G) of paired condition differences; test conditions are the sampling units.",
            "source_truth_sha256": truth_hash, "records": paired,
        })
        with (output / "paired_score_differences.csv").open("w", newline="") as handle:
            if paired:
                writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
                writer.writeheader()
                writer.writerows(paired)
        return summaries


if __name__ == "__main__":
    main()
