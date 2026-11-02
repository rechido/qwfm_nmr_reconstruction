# ruff: noqa: E402

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.utils.classical_baseline_utils import (
    BAYES_BETA_SOURCES,
    CLASSICAL_BASELINE_PROTOCOL_VERSION,
    CLASSICAL_METHODS,
    ClassicalBaselineResult,
    solve_classical_baseline_batch,
)
from codes.data import ContinuousNMRSpec
from codes.evaluate.artifact_contract import classical_representation_metadata
from codes.utils.io_utils import atomic_torch_save, atomic_write_json, load_torch_artifact
from codes.utils.continuous_experiment_utils import (
    add_continuous_dataset_arguments,
    spec_from_args,
)


CLASSICAL_SAMPLING_ARTIFACT_SCHEMA_VERSION = 2
CLASSICAL_SOLVER_RESULT_SCHEMA_VERSION = "classical_solver_results_v1"


def _optional_float_array(
    results: list[ClassicalBaselineResult], field: str
) -> np.ndarray:
    return np.asarray(
        [
            np.nan if getattr(result, field) is None else float(getattr(result, field))
            for result in results
        ],
        dtype=np.float64,
    )


def _pack_classical_results(
    results: list[ClassicalBaselineResult], *, bayes_beta_source: str | None
) -> dict:
    """Pack the unchanged legacy result objects into a posterior artifact."""

    if not results:
        raise ValueError("At least one classical result is required.")
    bins = int(results[0].spectrum.shape[0])
    if any(result.spectrum.shape != (bins,) for result in results):
        raise ValueError("Classical solver results have inconsistent spectrum shapes.")
    return {
        "schema_version": CLASSICAL_SOLVER_RESULT_SCHEMA_VERSION,
        "solver_protocol_version": CLASSICAL_BASELINE_PROTOCOL_VERSION,
        "method": results[0].method,
        "bayes_beta_source": bayes_beta_source,
        "condition_count": len(results),
        "num_spectrum_bins": bins,
        "raw_spectra": np.stack([result.raw_spectrum for result in results]),
        "normalized_spectra": np.stack([result.spectrum for result in results]),
        "diagnostics": {
            "regularization_parameter": _optional_float_array(results, "regularization_parameter"),
            "raw_mass": _optional_float_array(results, "raw_mass"),
            "raw_forward_sse": _optional_float_array(results, "raw_forward_sse"),
            "normalized_forward_sse": _optional_float_array(results, "normalized_forward_sse"),
            "unregularized_forward_sse": _optional_float_array(results, "unregularized_forward_sse"),
            "target_forward_sse": _optional_float_array(results, "target_forward_sse"),
            "achieved_residual_ratio": _optional_float_array(results, "achieved_residual_ratio"),
            "beta": _optional_float_array(results, "beta"),
            "objective_value": _optional_float_array(results, "objective_value"),
            "objective_evaluations": np.asarray([result.objective_evaluations for result in results], dtype=np.int64),
            "optimizer_success": np.asarray([result.optimizer_success for result in results], dtype=np.bool_),
            "selection_status": np.asarray([result.selection_status for result in results], dtype=np.str_),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one classical NMR baseline.")
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", choices=CLASSICAL_METHODS, required=True)
    add_continuous_dataset_arguments(parser)
    parser.add_argument(
        "--bayes_beta_source", choices=BAYES_BETA_SOURCES, default="official_nnls_residual"
    )
    parser.add_argument("--num_samples_per_condition", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--x2_factor", type=float, default=1.02)
    parser.add_argument("--regularization_minimum", type=float, default=1.0e-8)
    parser.add_argument("--regularization_maximum", type=float, default=None)
    parser.add_argument("--optimizer_tolerance", type=float, default=1.0e-5)
    parser.add_argument("--optimizer_max_iterations", type=int, default=300)
    parser.add_argument("--nnls_max_iterations", type=int, default=0)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--overwrite", action="store_true")
    cache_group.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args(argv)
    if args.num_samples_per_condition <= 0 or args.num_workers <= 0:
        parser.error("Sample and worker counts must be positive.")
    if args.nnls_max_iterations < 0:
        parser.error("nnls_max_iterations must be nonnegative.")
    if args.regularization_maximum is not None and (
        args.regularization_maximum <= args.regularization_minimum
    ):
        parser.error(
            "regularization_maximum must exceed regularization_minimum."
        )
    return args


def _finite_mean(values: np.ndarray) -> float | None:
    valid = values[np.isfinite(values)]
    return None if valid.size == 0 else float(valid.mean())


def _nested_equal(first, second) -> bool:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and tuple(first.shape) == tuple(second.shape)
            and bool(torch.equal(first.detach().cpu(), second.detach().cpu()))
        )
    if isinstance(first, dict) or isinstance(second, dict):
        return (
            isinstance(first, dict)
            and isinstance(second, dict)
            and set(first) == set(second)
            and all(_nested_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second)
            and all(_nested_equal(left, right) for left, right in zip(first, second))
        )
    return type(first) is type(second) and first == second


def validate_cached_classical_artifact(
    artifact,
    *,
    reference,
    spec: ContinuousNMRSpec,
    method_name: str,
    solver_method: str,
    bayes_beta_source: str | None,
    solver_options: dict,
    num_samples_per_condition: int,
) -> None:
    """Validate a deterministic solver cache against the full invocation."""

    if not isinstance(artifact, dict):
        raise ValueError("Cached classical samples.pt must contain a dictionary.")
    expected_scalars = {
        "artifact_schema_version": CLASSICAL_SAMPLING_ARTIFACT_SCHEMA_VERSION,
        "protocol_id": spec.protocol_id,
        "specification_hash": spec.specification_hash,
        "method": method_name,
        "num_samples_per_condition": num_samples_per_condition,
    }
    for field, expected in expected_scalars.items():
        if artifact.get(field) != expected:
            raise ValueError(
                f"Cached classical artifact has stale or invalid {field!r}: "
                f"expected {expected!r}, got {artifact.get(field)!r}."
            )
    expected_nested = {
        "specification": spec.to_dict(),
        "condition_ids": reference.get("condition_ids"),
        "measurements": reference.get("measurements"),
        "grid": reference.get("grid"),
        "representation_metadata": classical_representation_metadata(),
        "sampling_config": {
            "dirac_replication_count": num_samples_per_condition,
            "solver_options": solver_options,
        },
    }
    for field, expected in expected_nested.items():
        if field not in artifact or not _nested_equal(artifact[field], expected):
            raise ValueError(
                f"Cached classical artifact {field!r} does not match the invocation."
            )
    groups = int(reference["condition_ids"].shape[0])
    bins = spec.num_spectrum_bins
    samples = artifact.get("samples")
    expected_shape = (groups, num_samples_per_condition, bins)
    if not isinstance(samples, torch.Tensor) or tuple(samples.shape) != expected_shape:
        raise ValueError(f"Cached classical samples must have shape {expected_shape}.")
    if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
        raise ValueError("Cached classical samples must be finite and floating point.")
    if bool((samples < 0).any()) or float(
        (samples.sum(dim=-1) - 1.0).abs().max()
    ) > 2.0e-4:
        raise ValueError("Cached classical samples violate the simplex contract.")
    if not bool(torch.equal(samples, samples[:, :1].expand_as(samples))):
        raise ValueError("Cached classical Dirac replicas are not identical.")
    forward_spectra = artifact.get("forward_spectra")
    forward_count = 1
    if (
        not isinstance(forward_spectra, torch.Tensor)
        or tuple(forward_spectra.shape) != (groups, forward_count, bins)
        or not forward_spectra.is_floating_point()
        or not bool(torch.isfinite(forward_spectra).all())
        or bool((forward_spectra < 0).any())
    ):
        raise ValueError("Cached classical forward_spectra are invalid.")
    unique_counts = artifact.get("generated_unique_count")
    if not isinstance(unique_counts, torch.Tensor) or not bool(
        torch.equal(unique_counts.cpu(), torch.ones(groups, dtype=torch.int64))
    ):
        raise ValueError("Cached classical generated_unique_count is invalid.")
    packed = artifact.get("classical_results")
    if not isinstance(packed, dict) or packed.get("schema_version") != (
        CLASSICAL_SOLVER_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("Cached classical_results schema is invalid.")
    expected_packed_fields = {
        "schema_version",
        "solver_protocol_version",
        "method",
        "bayes_beta_source",
        "condition_count",
        "num_spectrum_bins",
        "raw_spectra",
        "normalized_spectra",
        "diagnostics",
    }
    if set(packed) != expected_packed_fields:
        raise ValueError("Cached classical_results fields are incomplete or unknown.")
    if (
        packed["solver_protocol_version"] != CLASSICAL_BASELINE_PROTOCOL_VERSION
        or packed["method"] != solver_method
        or packed["bayes_beta_source"] != bayes_beta_source
        or packed["condition_count"] != groups
        or packed["num_spectrum_bins"] != bins
    ):
        raise ValueError("Cached classical_results solver identity is invalid.")
    raw = np.asarray(packed.get("raw_spectra"))
    normalized = np.asarray(packed.get("normalized_spectra"))
    if raw.shape != (groups, bins) or normalized.shape != (groups, bins):
        raise ValueError("Cached classical solver spectra have invalid shapes.")
    if not np.array_equal(
        raw.astype(np.float32), forward_spectra[:, 0].cpu().numpy()
    ):
        raise ValueError("Cached raw solver spectra disagree with forward_spectra.")
    if not np.array_equal(
        normalized.astype(np.float32), samples[:, 0].cpu().numpy()
    ):
        raise ValueError("Cached normalized solver spectra disagree with samples.")
    diagnostics = packed["diagnostics"]
    expected_diagnostics = {
        "regularization_parameter",
        "raw_mass",
        "raw_forward_sse",
        "normalized_forward_sse",
        "unregularized_forward_sse",
        "target_forward_sse",
        "achieved_residual_ratio",
        "beta",
        "objective_value",
        "objective_evaluations",
        "optimizer_success",
        "selection_status",
    }
    if not isinstance(diagnostics, dict) or set(diagnostics) != expected_diagnostics:
        raise ValueError("Cached classical solver diagnostics schema is invalid.")
    if any(
        np.asarray(value).shape != (groups,)
        for value in diagnostics.values()
    ):
        raise ValueError("Cached classical solver diagnostics have invalid shapes.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    destination = output / "samples.pt"
    if destination.exists() and not args.overwrite and not args.reuse_existing:
        raise FileExistsError(
            f"Sampling output already exists: {destination}. Pass --reuse_existing "
            "to validate and reuse it, or --overwrite to replace it."
        )
    reference = load_torch_artifact(args.reference_path)
    spec = ContinuousNMRSpec.from_dict(reference["specification"])
    if reference.get("specification_hash") != spec.specification_hash:
        raise ValueError("Reference specification payload and hash disagree.")
    requested_spec = spec_from_args(args)
    if requested_spec.specification_hash != spec.specification_hash:
        raise ValueError(
            "The explicit continuous dataset arguments do not match the "
            "reference artifact specification."
        )
    kernel = reference["grid"]["kernel"].detach().cpu().numpy()
    measurements = reference["measurements"].detach().cpu().numpy()
    options = {
        "x2_factor": args.x2_factor,
        "bayes_beta_source": args.bayes_beta_source,
        "regularization_minimum": args.regularization_minimum,
        "optimizer_tolerance": args.optimizer_tolerance,
        "optimizer_max_iterations": args.optimizer_max_iterations,
        "nnls_max_iterations": args.nnls_max_iterations,
    }
    if args.regularization_maximum is not None:
        options["regularization_maximum"] = args.regularization_maximum
    suffix = (
        f"_{args.bayes_beta_source}" if args.method == "bayesreg_i" else ""
    )
    method_name = f"classical_{args.method}{suffix}"
    if destination.exists() and args.reuse_existing:
        validate_cached_classical_artifact(
            load_torch_artifact(destination),
            reference=reference,
            spec=spec,
            method_name=method_name,
            solver_method=args.method,
            bayes_beta_source=(
                args.bayes_beta_source if args.method == "bayesreg_i" else None
            ),
            solver_options=options,
            num_samples_per_condition=args.num_samples_per_condition,
        )
        print(destination)
        return
    started = time.perf_counter()
    results = solve_classical_baseline_batch(
        kernel,
        measurements,
        method=args.method,
        noise_standard_deviation=spec.noise_standard_deviation,
        num_workers=args.num_workers,
        **options,
    )
    elapsed = time.perf_counter() - started
    packed = _pack_classical_results(
        results,
        bayes_beta_source=(
            args.bayes_beta_source if args.method == "bayesreg_i" else None
        ),
    )
    forward_spectra = torch.from_numpy(packed["raw_spectra"]).to(torch.float32)[
        :, None, :
    ]
    point_predictions = torch.from_numpy(packed["normalized_spectra"]).to(torch.float32)
    samples = point_predictions[:, None, :].expand(
        -1, args.num_samples_per_condition, -1
    ).clone()
    artifact = {
        "artifact_schema_version": CLASSICAL_SAMPLING_ARTIFACT_SCHEMA_VERSION,
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "condition_ids": reference["condition_ids"].clone(),
        "measurements": reference["measurements"].clone(),
        "samples": samples,
        "forward_spectra": forward_spectra,
        "representation_metadata": classical_representation_metadata(),
        "grid": reference["grid"],
        "method": method_name,
        "num_samples_per_condition": args.num_samples_per_condition,
        "reference_path": str(Path(args.reference_path).expanduser().resolve()),
        "sampling_config": {
            "dirac_replication_count": args.num_samples_per_condition,
            "solver_options": options,
        },
        "generated_unique_count": torch.ones(samples.shape[0], dtype=torch.int64),
        "classical_results": packed,
        "sampling_wall_seconds": elapsed,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(destination, artifact)
    diagnostics = packed["diagnostics"]
    atomic_write_json(
        output / "sampling_summary.json",
        {
            "protocol_id": spec.protocol_id,
            "specification_hash": spec.specification_hash,
            "method": method_name,
            "sampling_wall_seconds": elapsed,
            "condition_count": int(samples.shape[0]),
            "num_samples_per_condition": args.num_samples_per_condition,
            "selection_status_counts": dict(
                collections.Counter(diagnostics["selection_status"])
            ),
            "optimizer_success_fraction": float(
                np.asarray(diagnostics["optimizer_success"], dtype=np.float64).mean()
            ),
            "mean_regularization_parameter": _finite_mean(
                diagnostics["regularization_parameter"]
            ),
            "mean_raw_forward_sse": _finite_mean(diagnostics["raw_forward_sse"]),
            "mean_normalized_forward_sse": _finite_mean(
                diagnostics["normalized_forward_sse"]
            ),
            "mean_raw_mass": _finite_mean(diagnostics["raw_mass"]),
        },
    )
    print(destination)


if __name__ == "__main__":
    main()
