# ruff: noqa: E402
"""Offline evaluation of cached conditional posterior samples.

This entrypoint never loads a network or performs a forward pass.  It compares
one common ``samples.pt`` artifact with a primary SMC reference run and an
independent oracle SMC run.  Reference clouds are deterministically subsampled
to the generated sample count for repeated, condition-wise evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.evaluate.artifact_contract import (
    NORMALIZED_HISTOGRAM_REPRESENTATION,
    RAW_FORWARD_SPECTRUM_REPRESENTATION,
    validate_optional_forward_spectra,
)
from codes.metrics import (
    GROUND_METRICS,
    METRIC_PROTOCOL_VERSION,
    compute_energy_score_report,
    compute_kernel_score_report,
    compute_sample_set_metrics,
    derived_quantities,
    forward_residual_diagnostics,
    sample_set_diagnostics,
    summarize_derived_quantities,
)
from codes.utils.io_utils import atomic_write_json, load_torch_artifact


RAW_FILENAME = "evaluation_raw.json"
SUMMARY_FILENAME = "evaluation_summary.json"
EVALUATION_PROTOCOL_VERSION = "saved_samples_reference_oracle_v1"
_GRID_TENSOR_FIELDS = (
    "log_rate_edges",
    "rate_edges",
    "rate_centers",
    "echo_times_64",
    "echo_subset_indices",
    "echo_times",
    "kernel",
)


@dataclass(frozen=True)
class EvaluationSettings:
    sample_count: int = 256
    reference_repeats: int = 20
    nearest_k: int = 3
    w2_num_quantiles: int = 2048
    distance_chunk_size: int = 64
    seed: int = 2026090304
    credible_interval_level: float = 0.90
    kernel_bandwidth: float | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "sample_count",
            "reference_repeats",
            "nearest_k",
            "w2_num_quantiles",
            "distance_chunk_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.sample_count <= self.nearest_k:
            raise ValueError("sample_count must exceed nearest_k for PRDC.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")
        if not 0.0 < float(self.credible_interval_level) < 1.0:
            raise ValueError("credible_interval_level must lie in (0,1).")
        if self.kernel_bandwidth is not None and (
            isinstance(self.kernel_bandwidth, bool)
            or not math.isfinite(float(self.kernel_bandwidth))
            or self.kernel_bandwidth <= 0
        ):
            raise ValueError("kernel_bandwidth must be finite and positive, or None.")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _required(payload: dict[str, Any], key: str, artifact_name: str) -> Any:
    if key not in payload:
        raise ValueError(f"{artifact_name} is missing required field {key!r}.")
    return payload[key]


def _tensor_equal(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    floating_atol: float,
) -> bool:
    if tuple(first.shape) != tuple(second.shape):
        return False
    if first.is_floating_point() or second.is_floating_point():
        return bool(
            torch.allclose(
                first.detach().to(device="cpu", dtype=torch.float64),
                second.detach().to(device="cpu", dtype=torch.float64),
                rtol=0.0,
                atol=floating_atol,
            )
        )
    return bool(torch.equal(first.detach().cpu(), second.detach().cpu()))


def _validate_simplex_tensor(samples: torch.Tensor, *, artifact_name: str) -> None:
    """Validate a potentially large bank without an expensive uniqueness sort."""

    if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
        raise ValueError(f"{artifact_name} must contain finite floating-point samples.")
    if bool((samples < 0).any()):
        raise ValueError(f"{artifact_name} contains negative histogram mass.")
    masses = samples.sum(dim=-1)
    if not bool(
        torch.allclose(masses, torch.ones_like(masses), atol=1.0e-5, rtol=1.0e-5)
    ):
        raise ValueError(f"{artifact_name} contains a non-unit histogram.")


def _validate_grid(
    grid: dict[str, Any],
    expected_grid: dict[str, Any],
    *,
    artifact_name: str,
) -> None:
    if not isinstance(grid, dict):
        raise ValueError(f"{artifact_name}.grid must be a dictionary.")
    expected_hash = expected_grid["specification_hash"]
    if grid.get("specification_hash") != expected_hash:
        raise ValueError(f"{artifact_name}.grid specification hash differs.")
    for field in _GRID_TENSOR_FIELDS:
        value = _required(grid, field, f"{artifact_name}.grid")
        expected = expected_grid[field]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{artifact_name}.grid[{field!r}] must be a tensor.")
        tolerance = 0.0 if not expected.is_floating_point() else 1.0e-12
        if not _tensor_equal(value, expected, floating_atol=tolerance):
            raise ValueError(f"{artifact_name}.grid[{field!r}] differs from the spec.")


def _validate_common_identity(
    payload: dict[str, Any],
    *,
    artifact_name: str,
    spec: ContinuousNMRSpec,
) -> None:
    if payload.get("protocol_id") != spec.protocol_id:
        raise ValueError(f"{artifact_name} uses a different protocol_id.")
    if payload.get("specification_hash") != spec.specification_hash:
        raise ValueError(f"{artifact_name} uses a different specification hash.")
    if "specification" in payload:
        local_spec = ContinuousNMRSpec.from_dict(payload["specification"])
        if local_spec.specification_hash != spec.specification_hash:
            raise ValueError(f"{artifact_name} embeds an inconsistent specification.")


def validate_evaluation_artifacts(
    samples_artifact: dict[str, Any],
    primary_reference: dict[str, Any],
    oracle_reference: dict[str, Any],
    *,
    sample_count: int,
) -> ContinuousNMRSpec:
    """Validate every shared identity field before any metric is computed."""

    for payload, name in (
        (samples_artifact, "samples artifact"),
        (primary_reference, "primary reference"),
        (oracle_reference, "oracle reference"),
    ):
        if not isinstance(payload, dict):
            raise ValueError(f"{name} must be a dictionary artifact.")
    specification = _required(primary_reference, "specification", "primary reference")
    if not isinstance(specification, dict):
        raise ValueError("primary reference specification must be a dictionary.")
    spec = ContinuousNMRSpec.from_dict(specification)
    for payload, name in (
        (samples_artifact, "samples artifact"),
        (primary_reference, "primary reference"),
        (oracle_reference, "oracle reference"),
    ):
        _validate_common_identity(payload, artifact_name=name, spec=spec)

    primary_run = _required(primary_reference, "reference_run_id", "primary reference")
    oracle_run = _required(oracle_reference, "reference_run_id", "oracle reference")
    if (
        not isinstance(primary_run, int)
        or isinstance(primary_run, bool)
        or primary_run < 0
        or not isinstance(oracle_run, int)
        or isinstance(oracle_run, bool)
        or oracle_run < 0
    ):
        raise ValueError("reference_run_id must be a nonnegative integer.")
    if primary_run == oracle_run:
        raise ValueError("primary and oracle references must have distinct run IDs.")
    primary_smc_settings = _required(
        primary_reference, "smc_settings", "primary reference"
    )
    oracle_smc_settings = _required(
        oracle_reference, "smc_settings", "oracle reference"
    )
    if primary_smc_settings != oracle_smc_settings:
        raise ValueError("primary and oracle references use different SMC settings.")
    for reference, name in (
        (primary_reference, "primary reference"),
        (oracle_reference, "oracle reference"),
    ):
        master_seed = _required(reference, "reference_master_seed", name)
        if (
            not isinstance(master_seed, int)
            or isinstance(master_seed, bool)
            or master_seed < 0
        ):
            raise ValueError(f"{name} reference_master_seed must be nonnegative.")

    expected_grid = build_nmr_grid(spec).as_artifact()
    for payload, name in (
        (samples_artifact, "samples artifact"),
        (primary_reference, "primary reference"),
        (oracle_reference, "oracle reference"),
    ):
        _validate_grid(
            _required(payload, "grid", name),
            expected_grid,
            artifact_name=name,
        )

    condition_ids = _required(samples_artifact, "condition_ids", "samples artifact")
    measurements = _required(samples_artifact, "measurements", "samples artifact")
    samples = _required(samples_artifact, "samples", "samples artifact")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (condition_ids, measurements, samples)
    ):
        raise ValueError(
            "sample condition_ids, measurements, and samples must be tensors."
        )
    if condition_ids.ndim != 1 or measurements.ndim != 2 or samples.ndim != 3:
        raise ValueError("sample fields must have shapes [G], [G,M], and [G,K,N].")
    groups = int(condition_ids.shape[0])
    if groups < 1 or int(torch.unique(condition_ids).numel()) != groups:
        raise ValueError("condition_ids must be nonempty and unique.")
    if tuple(measurements.shape) != (groups, spec.num_measurements):
        raise ValueError("samples artifact measurements have the wrong shape.")
    if samples.shape[0] != groups or samples.shape[2] != spec.num_spectrum_bins:
        raise ValueError("samples tensor has incompatible G or N dimensions.")
    if samples.shape[1] < sample_count:
        raise ValueError(
            f"samples artifact contains K={samples.shape[1]}, fewer than {sample_count}."
        )
    _validate_simplex_tensor(samples, artifact_name="samples artifact")
    validate_optional_forward_spectra(
        samples_artifact,
        groups=groups,
        num_bins=spec.num_spectrum_bins,
        artifact_name="samples artifact",
    )

    for reference, name in (
        (primary_reference, "primary reference"),
        (oracle_reference, "oracle reference"),
    ):
        reference_ids = _required(reference, "condition_ids", name)
        reference_measurements = _required(reference, "measurements", name)
        histograms = _required(reference, "reference_histograms", name)
        if not all(
            isinstance(value, torch.Tensor)
            for value in (reference_ids, reference_measurements, histograms)
        ):
            raise ValueError(f"{name} identity/sample fields must be tensors.")
        if not _tensor_equal(condition_ids, reference_ids, floating_atol=0.0):
            raise ValueError(f"{name} condition IDs differ from the samples artifact.")
        if not _tensor_equal(
            measurements, reference_measurements, floating_atol=1.0e-7
        ):
            raise ValueError(f"{name} measurements differ from the samples artifact.")
        if histograms.ndim != 3 or tuple(histograms.shape[::2]) != (
            groups,
            spec.num_spectrum_bins,
        ):
            raise ValueError(f"{name} reference_histograms has the wrong shape.")
        if histograms.shape[1] < sample_count:
            raise ValueError(f"{name} has fewer than {sample_count} posterior samples.")
        _validate_simplex_tensor(histograms, artifact_name=name)

    primary_truth = _required(
        primary_reference, "truth_histograms", "primary reference"
    )
    oracle_truth = _required(
        oracle_reference, "truth_histograms", "oracle reference"
    )
    if not isinstance(primary_truth, torch.Tensor) or not isinstance(
        oracle_truth, torch.Tensor
    ):
        raise ValueError("reference truth_histograms must be tensors.")
    if tuple(primary_truth.shape) != (groups, spec.num_spectrum_bins):
        raise ValueError("primary truth_histograms has the wrong shape.")
    if not _tensor_equal(primary_truth, oracle_truth, floating_atol=1.0e-7):
        raise ValueError("primary and oracle references do not share test truths.")
    return spec


def _keyed_subset_indices(
    population: int,
    sample_count: int,
    *,
    seed: int,
    specification_hash: str,
    run_id: int,
    condition_id: int,
    repeat: int,
    role: str,
) -> torch.Tensor:
    key = (
        f"{EVALUATION_PROTOCOL_VERSION}|{seed}|{specification_hash}|{run_id}|"
        f"{condition_id}|{repeat}|{role}"
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    local_seed = int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    return torch.randperm(population, generator=generator)[:sample_count]


def _empirical_scalar_distances(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("derived sample values must be vectors.")
    if reference.numel() != candidate.numel():
        raise ValueError("derived empirical W1 requires matched sample counts.")
    reference = reference.to(torch.float64)
    candidate = candidate.to(torch.float64)
    empirical_w1 = (reference.sort().values - candidate.sort().values).abs().mean()
    cross = (reference[:, None] - candidate[None, :]).abs().mean()
    within_reference = (reference[:, None] - reference[None, :]).abs().mean()
    within_candidate = (candidate[:, None] - candidate[None, :]).abs().mean()
    energy = (2.0 * cross - within_reference - within_candidate).clamp_min(0.0)
    return float(empirical_w1.item()), float(energy.item())


def _derived_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    truth: torch.Tensor,
    log_rate_edges: torch.Tensor,
    *,
    interval_level: float,
) -> dict[str, dict[str, float]]:
    reference_values = derived_quantities(reference, log_rate_edges)
    candidate_values = derived_quantities(candidate, log_rate_edges)
    truth_values = derived_quantities(truth[None, :], log_rate_edges)
    reference_summary = summarize_derived_quantities(reference_values)
    candidate_summary = summarize_derived_quantities(candidate_values)
    alpha = 0.5 * (1.0 - interval_level)
    result: dict[str, dict[str, float]] = {}
    for name in reference_values:
        ref = reference_values[name].to(torch.float64)
        cand = candidate_values[name].to(torch.float64)
        true_value = float(truth_values[name][0].item())
        candidate_bounds = torch.quantile(
            cand,
            torch.tensor([alpha, 1.0 - alpha], device=cand.device, dtype=torch.float64),
        )
        reference_bounds = torch.quantile(
            ref,
            torch.tensor([alpha, 1.0 - alpha], device=ref.device, dtype=torch.float64),
        )
        empirical_w1, energy = _empirical_scalar_distances(ref, cand)
        quantile_absolute_error = (
            sum(
                abs(candidate_summary[name][key] - reference_summary[name][key])
                for key in ("q05", "median", "q95")
            )
            / 3.0
        )
        result[name] = {
            "reference_mean": reference_summary[name]["mean"],
            "candidate_mean": candidate_summary[name]["mean"],
            "absolute_mean_error": abs(
                candidate_summary[name]["mean"] - reference_summary[name]["mean"]
            ),
            "quantile_absolute_error_mean": quantile_absolute_error,
            "empirical_w1": empirical_w1,
            "energy_distance": energy,
            "truth_value": true_value,
            "truth_in_candidate_interval": float(
                candidate_bounds[0].item() <= true_value <= candidate_bounds[1].item()
            ),
            "candidate_interval_width": float(
                (candidate_bounds[1] - candidate_bounds[0]).item()
            ),
            "truth_in_reference_interval": float(
                reference_bounds[0].item() <= true_value <= reference_bounds[1].item()
            ),
            "reference_interval_width": float(
                (reference_bounds[1] - reference_bounds[0]).item()
            ),
        }
    return result


def _evaluate_candidate(
    primary: torch.Tensor,
    candidate: torch.Tensor,
    forward_candidate: torch.Tensor,
    truth: torch.Tensor,
    measurement: torch.Tensor,
    log_rate_edges: torch.Tensor,
    kernel: torch.Tensor,
    *,
    spec: ContinuousNMRSpec,
    settings: EvaluationSettings,
    forward_requires_unit_mass: bool,
) -> dict[str, Any]:
    ground = {
        metric: compute_sample_set_metrics(
            primary,
            candidate,
            metric=metric,
            log_rate_edges=log_rate_edges,
            nearest_k=settings.nearest_k,
            num_quantiles=settings.w2_num_quantiles,
            chunk_size=settings.distance_chunk_size,
        )
        for metric in GROUND_METRICS
    }
    return {
        "ground_metrics": ground,
        "forward_residual": forward_residual_diagnostics(
            forward_candidate,
            measurement,
            kernel,
            noise_standard_deviation=spec.noise_standard_deviation,
            require_unit_mass=forward_requires_unit_mass,
        ),
        "derived_metrics": _derived_comparison(
            primary,
            candidate,
            truth,
            log_rate_edges,
            interval_level=settings.credible_interval_level,
        ),
        "sample_diagnostics": sample_set_diagnostics(candidate),
    }


def _difference_tree(first: Any, second: Any) -> Any:
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            raise ValueError("model and oracle metric schemas differ.")
        return {key: _difference_tree(first[key], second[key]) for key in first}
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        value = float(first) - float(second)
        if not math.isfinite(value):
            raise ValueError("a model-minus-oracle metric is non-finite.")
        return value
    raise TypeError(
        "metric comparison trees may contain only dictionaries and scalars."
    )


def _comparison_projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ground_metrics": result["ground_metrics"],
        "forward_residual": result["forward_residual"],
        "derived_metrics": result["derived_metrics"],
    }


def _numeric_leaves(
    tree: Any, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], float]:
    if isinstance(tree, dict):
        result: dict[tuple[str, ...], float] = {}
        for key, value in tree.items():
            result.update(_numeric_leaves(value, (*prefix, str(key))))
        return result
    if isinstance(tree, (int, float)) and not isinstance(tree, bool):
        value = float(tree)
        if not math.isfinite(value):
            raise ValueError(f"non-finite aggregate leaf at {'.'.join(prefix)}.")
        return {prefix: value}
    return {}


def _assign_nested(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = tree
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def _aggregate_trees(trees: list[dict[str, Any]]) -> dict[str, Any]:
    if not trees:
        raise ValueError("cannot aggregate an empty metric collection.")
    flattened = [_numeric_leaves(tree) for tree in trees]
    paths = tuple(flattened[0].keys())
    if any(tuple(item.keys()) != paths for item in flattened[1:]):
        raise ValueError("metric schemas differ across evaluation records.")
    result: dict[str, Any] = {}
    for path in paths:
        values = torch.tensor([item[path] for item in flattened], dtype=torch.float64)
        _assign_nested(
            result,
            path,
            {
                "mean": float(values.mean().item()),
                "standard_deviation": float(values.std(unbiased=False).item()),
                "minimum": float(values.min().item()),
                "maximum": float(values.max().item()),
            },
        )
    return result


def evaluate_saved_samples_artifacts(
    samples_artifact: dict[str, Any],
    primary_reference: dict[str, Any],
    oracle_reference: dict[str, Any],
    *,
    settings: EvaluationSettings = EvaluationSettings(),
    condition_indices: list[int] | None = None,
    device: torch.device | str = "cpu",
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate already-generated samples and return raw and summary payloads."""

    spec = validate_evaluation_artifacts(
        samples_artifact,
        primary_reference,
        oracle_reference,
        sample_count=settings.sample_count,
    )
    groups = int(samples_artifact["condition_ids"].shape[0])
    selected = (
        list(range(groups)) if condition_indices is None else list(condition_indices)
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("condition_indices must be nonempty and unique.")
    if any(index < 0 or index >= groups for index in selected):
        raise ValueError("condition_indices contains an out-of-range value.")
    resolved_device = torch.device(device)
    # A proper score against each simulator truth, computed once per condition.
    # Independent SMC subsampling repeats do not create additional test truths.
    energy_score = compute_energy_score_report(
        samples_artifact,
        primary_reference,
        sample_count=settings.sample_count,
        condition_indices=selected,
        distance_chunk_size=settings.distance_chunk_size,
        device=resolved_device,
    )
    kernel_score = compute_kernel_score_report(
        samples_artifact,
        primary_reference,
        sample_count=settings.sample_count,
        condition_indices=selected,
        distance_chunk_size=settings.distance_chunk_size,
        device=resolved_device,
        bandwidth=settings.kernel_bandwidth,
    )
    grid = primary_reference["grid"]
    log_rate_edges = grid["log_rate_edges"].to(resolved_device)
    kernel = grid["kernel"].to(resolved_device)
    primary_run_id = int(primary_reference["reference_run_id"])
    oracle_run_id = int(oracle_reference["reference_run_id"])
    raw_records: list[dict[str, Any]] = []
    optional_forward_bank = validate_optional_forward_spectra(
        samples_artifact,
        groups=groups,
        num_bins=spec.num_spectrum_bins,
        artifact_name="samples artifact",
    )
    model_forward_representation = (
        RAW_FORWARD_SPECTRUM_REPRESENTATION
        if optional_forward_bank is not None
        else NORMALIZED_HISTOGRAM_REPRESENTATION
    )

    for condition_index in selected:
        condition_id = int(samples_artifact["condition_ids"][condition_index])
        measurement = samples_artifact["measurements"][condition_index].to(
            resolved_device
        )
        truth = primary_reference["truth_histograms"][condition_index].to(
            resolved_device
        )
        generated = samples_artifact["samples"][
            condition_index, : settings.sample_count
        ].to(resolved_device)
        generated_forward = (
            generated
            if optional_forward_bank is None
            else optional_forward_bank[condition_index].to(resolved_device)
        )
        primary_bank = primary_reference["reference_histograms"][condition_index]
        oracle_bank = oracle_reference["reference_histograms"][condition_index]
        for repeat in range(settings.reference_repeats):
            primary_indices = _keyed_subset_indices(
                int(primary_bank.shape[0]),
                settings.sample_count,
                seed=settings.seed,
                specification_hash=spec.specification_hash,
                run_id=primary_run_id,
                condition_id=condition_id,
                repeat=repeat,
                role="primary",
            )
            oracle_indices = _keyed_subset_indices(
                int(oracle_bank.shape[0]),
                settings.sample_count,
                seed=settings.seed,
                specification_hash=spec.specification_hash,
                run_id=oracle_run_id,
                condition_id=condition_id,
                repeat=repeat,
                role="oracle",
            )
            primary = primary_bank.index_select(
                0, primary_indices.to(primary_bank.device)
            ).to(resolved_device)
            oracle = oracle_bank.index_select(
                0, oracle_indices.to(oracle_bank.device)
            ).to(resolved_device)
            model_result = _evaluate_candidate(
                primary,
                generated,
                generated_forward,
                truth,
                measurement,
                log_rate_edges,
                kernel,
                spec=spec,
                settings=settings,
                forward_requires_unit_mass=optional_forward_bank is None,
            )
            oracle_result = _evaluate_candidate(
                primary,
                oracle,
                oracle,
                truth,
                measurement,
                log_rate_edges,
                kernel,
                spec=spec,
                settings=settings,
                forward_requires_unit_mass=True,
            )
            model_projection = _comparison_projection(model_result)
            oracle_projection = _comparison_projection(oracle_result)
            raw_records.append(
                {
                    "condition_index": condition_index,
                    "condition_id": condition_id,
                    "repeat_index": repeat,
                    "primary_reference_indices": primary_indices.tolist(),
                    "oracle_reference_indices": oracle_indices.tolist(),
                    "model": model_result,
                    "oracle": oracle_result,
                    "model_minus_oracle": _difference_tree(
                        model_projection, oracle_projection
                    ),
                }
            )
        if progress:
            print(
                f"evaluated condition {condition_index} "
                f"({selected.index(condition_index) + 1}/{len(selected)})",
                flush=True,
            )

    raw_payload = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "specification_hash": spec.specification_hash,
        "primary_reference_run_id": primary_run_id,
        "oracle_reference_run_id": oracle_run_id,
        "settings": settings.to_dict(),
        "representation_usage": {
            "distribution_prdc_and_derived": NORMALIZED_HISTOGRAM_REPRESENTATION,
            "model_forward_residual": model_forward_representation,
            "oracle_forward_residual": NORMALIZED_HISTOGRAM_REPRESENTATION,
        },
        "condition_indices": selected,
        "records": raw_records,
        "energy_score": energy_score,
        "kernel_score": kernel_score,
    }
    per_condition_summary = []
    for condition_index in selected:
        condition_records = [
            record
            for record in raw_records
            if record["condition_index"] == condition_index
        ]
        per_condition_summary.append(
            {
                "condition_index": condition_index,
                "condition_id": condition_records[0]["condition_id"],
                "repeat_count": len(condition_records),
                "aggregate": {
                    "model": _aggregate_trees(
                        [
                            _comparison_projection(record["model"])
                            for record in condition_records
                        ]
                    ),
                    "oracle": _aggregate_trees(
                        [
                            _comparison_projection(record["oracle"])
                            for record in condition_records
                        ]
                    ),
                    "model_minus_oracle": _aggregate_trees(
                        [record["model_minus_oracle"] for record in condition_records]
                    ),
                },
            }
        )
    summary_payload = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "specification_hash": spec.specification_hash,
        "primary_reference_run_id": primary_run_id,
        "oracle_reference_run_id": oracle_run_id,
        "settings": settings.to_dict(),
        "representation_usage": {
            "distribution_prdc_and_derived": NORMALIZED_HISTOGRAM_REPRESENTATION,
            "model_forward_residual": model_forward_representation,
            "oracle_forward_residual": NORMALIZED_HISTOGRAM_REPRESENTATION,
        },
        "condition_indices": selected,
        "record_count": len(raw_records),
        "energy_score": energy_score,
        "kernel_score": kernel_score,
        "per_condition": per_condition_summary,
        "aggregate": {
            "model": _aggregate_trees(
                [_comparison_projection(record["model"]) for record in raw_records]
            ),
            "oracle": _aggregate_trees(
                [_comparison_projection(record["oracle"]) for record in raw_records]
            ),
            "model_minus_oracle": _aggregate_trees(
                [record["model_minus_oracle"] for record in raw_records]
            ),
            "model_sample_diagnostics": _aggregate_trees(
                [record["model"]["sample_diagnostics"] for record in raw_records]
            ),
            "oracle_sample_diagnostics": _aggregate_trees(
                [record["oracle"]["sample_diagnostics"] for record in raw_records]
            ),
        },
        "metric_directions": {
            "precision_recall_density_coverage": "higher_is_better",
            "validity_distance_coverage_distance_energy_distance": "lower_is_better",
            "derived_distribution_discrepancy": "lower_is_better",
            "forward_residual": "diagnostic_not_forced_to_one",
            "interval_width": "context_dependent",
        },
    }
    return raw_payload, summary_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cached posterior samples without network inference."
    )
    parser.add_argument("--samples", required=True)
    parser.add_argument("--primary_reference", required=True)
    parser.add_argument("--oracle_reference", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--reference_repeats", type=int, default=20)
    parser.add_argument("--nearest_k", type=int, default=3)
    parser.add_argument("--w2_num_quantiles", type=int, default=2048)
    parser.add_argument("--distance_chunk_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026090304)
    parser.add_argument("--credible_interval_level", type=float, default=0.90)
    parser.add_argument("--kernel_bandwidth", type=float, default=None,
                        help="Fixed Gaussian bandwidth; default: independent simulator calibration median.")
    parser.add_argument("--condition_indices", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sample_path = Path(args.samples).resolve()
    primary_path = Path(args.primary_reference).resolve()
    oracle_path = Path(args.oracle_reference).resolve()
    if primary_path == oracle_path:
        raise ValueError("primary and oracle reference paths must be different.")
    output_dir = Path(args.output_dir).resolve()
    raw_path = output_dir / RAW_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    if not args.overwrite and (raw_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"evaluation output already exists in {output_dir}; pass --overwrite."
        )
    settings = EvaluationSettings(
        sample_count=args.sample_count,
        reference_repeats=args.reference_repeats,
        nearest_k=args.nearest_k,
        w2_num_quantiles=args.w2_num_quantiles,
        distance_chunk_size=args.distance_chunk_size,
        seed=args.seed,
        credible_interval_level=args.credible_interval_level,
        kernel_bandwidth=args.kernel_bandwidth,
    )
    raw, summary = evaluate_saved_samples_artifacts(
        load_torch_artifact(sample_path),
        load_torch_artifact(primary_path),
        load_torch_artifact(oracle_path),
        settings=settings,
        condition_indices=args.condition_indices,
        device=args.device,
        progress=True,
    )
    source_paths = {
        "samples": str(sample_path),
        "primary_reference": str(primary_path),
        "oracle_reference": str(oracle_path),
    }
    raw["source_paths"] = source_paths
    summary["source_paths"] = source_paths
    atomic_write_json(raw_path, raw)
    atomic_write_json(summary_path, summary)
    for score_name in ("energy_score", "kernel_score"):
        atomic_write_json(output_dir / f"{score_name}.json", summary[score_name])
    print(f"saved {raw_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
