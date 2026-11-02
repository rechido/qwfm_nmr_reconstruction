# ruff: noqa: E402
"""Run one classical point estimator on the independent proper-score test split."""

from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from codes.data import ContinuousNMRSpec
from codes.evaluate.artifact_contract import classical_representation_metadata
from codes.evaluate.evaluate_classical_baseline import (
    CLASSICAL_SAMPLING_ARTIFACT_SCHEMA_VERSION,
    _finite_mean,
    _pack_classical_results,
)
from codes.evaluate.generate_proper_score_samples import (
    PROTOCOL,
    _as_reference,
    validate_sample_cache,
)
from codes.utils.proper_score_conditions import load_proper_score_truth
from codes.utils.model_variant_utils import experiment_dimensions
from codes.utils.proper_score_locks import model_generation_lock
from codes.utils.classical_baseline_utils import (
    CLASSICAL_BASELINE_PROTOCOL_VERSION,
    solve_classical_baseline_batch,
)
from codes.utils.io_utils import atomic_write_json
from codes.utils.predictive_generation_utils import (
    _duration,
    _GenerationProgress,
    _save_cache,
    _verified_cache,
    file_sha256,
)


CLASSICAL_VARIANTS = {
    "classic_nnls": ("nnls", "official_nnls_residual"),
    "classic_x2_d2": ("x2_d2", "official_nnls_residual"),
    "classic_discrepancy_d2": ("discrepancy_d2", "official_nnls_residual"),
    "classic_bayesreg_i_official": ("bayesreg_i", "official_nnls_residual"),
    "classic_bayesreg_i_known_noise": ("bayesreg_i", "known_noise"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(CLASSICAL_VARIANTS), required=True)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--num_posterior_samples", type=int, default=8)
    parser.add_argument("--test_dataset_seed", type=int, default=2026092901)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--condition_batch_size", type=int, default=16)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--progress_interval_seconds", type=float, default=30.0)
    parser.add_argument("--x2_factor", type=float, default=1.02)
    parser.add_argument("--regularization_minimum", type=float, default=1.0e-8)
    parser.add_argument("--regularization_maximum", type=float, default=None)
    parser.add_argument("--optimizer_tolerance", type=float, default=1.0e-5)
    parser.add_argument("--optimizer_max_iterations", type=int, default=300)
    parser.add_argument("--nnls_max_iterations", type=int, default=0)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args(argv)
    if min(args.num_conditions, args.num_posterior_samples, args.num_workers,
           args.condition_batch_size, args.progress_every) < 1:
        parser.error("Condition, sample, batch, worker and progress counts must be positive.")
    if (not math.isfinite(args.progress_interval_seconds)
            or args.progress_interval_seconds <= 0 or args.nnls_max_iterations < 0):
        parser.error("Progress interval must be finite and positive; NNLS iterations must be nonnegative.")
    if not 0 <= args.test_dataset_seed < 2**63:
        parser.error("test_dataset_seed must lie in [0, 2**63).")
    if args.regularization_maximum is not None and (
        args.regularization_maximum <= args.regularization_minimum
    ):
        parser.error("regularization_maximum must exceed regularization_minimum.")
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    try:
        experiment_dimensions(args.experiment_dir)
    except ValueError as error:
        parser.error(str(error))
    args.run_dir = (args.run_dir or args.experiment_dir / "proper_scores" /
                    f"g{args.num_conditions}_k{args.num_posterior_samples}").expanduser().resolve()
    return args


def _load_conditions(args: argparse.Namespace) -> tuple[dict, ContinuousNMRSpec, Path]:
    path = args.run_dir / "test_conditions.pt"
    truth = load_proper_score_truth(
        args.run_dir, num_conditions=args.num_conditions,
        test_dataset_seed=args.test_dataset_seed,
    )
    spec = ContinuousNMRSpec.from_dict(truth["specification"])
    if (spec.num_measurements, spec.num_spectrum_bins) != experiment_dimensions(args.experiment_dir):
        raise ValueError("Shared truth simulator does not match the requested synthetic experiment.")
    return truth, spec, path


def _generate(args: argparse.Namespace) -> None:
    truth, spec, truth_path = _load_conditions(args)
    method, beta_source = CLASSICAL_VARIANTS[args.variant]
    options = {
        "x2_factor": args.x2_factor,
        "bayes_beta_source": beta_source,
        "regularization_minimum": args.regularization_minimum,
        "regularization_maximum": (
            args.regularization_maximum if args.regularization_maximum is not None
            else 2.0 if method == "bayesreg_i" else 10.0
        ),
        "optimizer_tolerance": args.optimizer_tolerance,
        "optimizer_max_iterations": args.optimizer_max_iterations,
        "nnls_max_iterations": args.nnls_max_iterations,
    }
    expected = {
        "protocol_version": PROTOCOL,
        "variant": args.variant,
        "num_conditions": args.num_conditions,
        "conditions_sha256": file_sha256(truth_path),
        "requested_sample_count": args.num_posterior_samples,
        "stored_sample_count": 1,
        "deterministic": True,
        "solver_protocol_version": CLASSICAL_BASELINE_PROTOCOL_VERSION,
        "solver_method": method,
        "bayes_beta_source": beta_source if method == "bayesreg_i" else None,
        "solver_options": options,
    }
    destination = args.run_dir / args.variant / "samples.pt"
    def record() -> None:
        atomic_write_json(destination.parent / "generation_manifest.json", {
            "protocol_version": PROTOCOL, "num_conditions": args.num_conditions,
            "conditions_file": str(truth_path),
            "conditions_sha256": expected["conditions_sha256"],
            "requested_sample_count": args.num_posterior_samples,
            "specification_hash": spec.specification_hash,
            "jobs": {args.variant: {
                "variant": args.variant, "samples_file": str(destination.relative_to(args.run_dir)),
                "samples_sha256": file_sha256(destination), "proper_score_provenance": expected,
            }},
        })

    cached = _verified_cache(destination, args.reuse_existing)
    if cached is not None:
        validate_sample_cache(cached, truth, expected)
        record()
        print(f"Reused checked classical point predictions: {destination}", flush=True)
        return

    reference = _as_reference(truth)
    kernel = reference["grid"]["kernel"].detach().cpu().numpy()
    measurements = reference["measurements"].detach().cpu().numpy()
    started = time.perf_counter()
    results = []
    progress = _GenerationProgress(SimpleNamespace(print=print), args.variant,
                                   args.num_conditions, args.progress_every,
                                   args.progress_interval_seconds)
    for step, start in enumerate(range(0, args.num_conditions, args.condition_batch_size), 1):
        stop = min(start + args.condition_batch_size, args.num_conditions)
        results.extend(solve_classical_baseline_batch(
            kernel, measurements[start:stop], method=method,
            noise_standard_deviation=spec.noise_standard_deviation,
            num_workers=args.num_workers, **options,
        ))
        progress.update(stop, step)
    elapsed = time.perf_counter() - started
    packed = _pack_classical_results(
        results, bayes_beta_source=beta_source if method == "bayesreg_i" else None,
    )
    samples = torch.from_numpy(packed["normalized_spectra"]).to(torch.float32)[:, None, :]
    artifact = {
        "artifact_schema_version": CLASSICAL_SAMPLING_ARTIFACT_SCHEMA_VERSION,
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "condition_ids": reference["condition_ids"].clone(),
        "measurements": reference["measurements"].clone(),
        "grid": reference["grid"],
        "method": f"classical_{method}" + (f"_{beta_source}" if method == "bayesreg_i" else ""),
        "samples": samples,
        "forward_spectra": torch.from_numpy(packed["raw_spectra"]).to(torch.float32)[:, None, :],
        "representation_metadata": classical_representation_metadata(),
        "deterministic": True,
        "num_samples_per_condition": 1,
        "reference_path": str(truth_path),
        "proper_score_provenance": expected,
        "sampling_config": {"solver_options": options},
        "generated_unique_count": torch.ones(args.num_conditions, dtype=torch.int64),
        "classical_results": packed,
        "sampling_wall_seconds": elapsed,
    }
    validate_sample_cache(artifact, truth, expected)
    _save_cache(destination, artifact)
    diagnostics = packed["diagnostics"]
    atomic_write_json(destination.parent / "sampling_summary.json", {
        "proper_score_provenance": expected,
        "condition_count": args.num_conditions,
        "num_samples_per_condition": 1,
        "deterministic": True,
        "sampling_wall_seconds": elapsed,
        "selection_status_counts": dict(collections.Counter(diagnostics["selection_status"])),
        "optimizer_success_fraction": float(np.asarray(diagnostics["optimizer_success"], dtype=float).mean()),
        "mean_regularization_parameter": _finite_mean(diagnostics["regularization_parameter"]),
        "mean_raw_forward_sse": _finite_mean(diagnostics["raw_forward_sse"]),
        "mean_normalized_forward_sse": _finite_mean(diagnostics["normalized_forward_sse"]),
    })
    record()
    print(f"Saved {destination} | total elapsed={_duration(time.perf_counter()-started)}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with model_generation_lock(args.run_dir, args.variant):
        _generate(args)


if __name__ == "__main__":
    main()
