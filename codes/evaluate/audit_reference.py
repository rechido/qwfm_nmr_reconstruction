# ruff: noqa: E402
"""Audit agreement between independently seeded SMC posterior references.

The audit is deliberately separate from model evaluation.  It asks whether
independent numerical approximations of the same conditional posterior agree
closely enough to be used as a reference floor.  Comparisons are performed per
measurement condition; samples from different conditions are never pooled.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.metrics import (
    GROUND_METRICS,
    compute_sample_set_metrics,
    derived_quantities,
    forward_residual_diagnostics,
    pairwise_distances,
    summarize_derived_quantities,
)
from codes.utils.nmr_utils import (
    histogram_to_quantiles,
    make_normalized_log_rate_edges,
    quantile_wasserstein_representation_signature,
    quantiles_to_histograms,
)
from codes.utils.duplicate_utils import exact_duplicate_diagnostics
from codes.utils.io_utils import atomic_write_json, load_torch_artifact


AUDIT_PROTOCOL_VERSION = "independent_smc_reference_audit_v3"
RAW_FILENAME = "reference_audit_raw.json"
SUMMARY_FILENAME = "reference_audit_summary.json"
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
class AuditSettings:
    sample_count: int = 256
    w2_num_quantiles: int = 2048
    distance_chunk_size: int = 64
    seed: int = 2026090305
    representation_quantiles: int = 128
    suitability_lower_quantile: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "sample_count",
            "w2_num_quantiles",
            "distance_chunk_size",
            "representation_quantiles",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.sample_count < 2:
            raise ValueError(
                "sample_count must be at least two for off-diagonal spread."
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")
        if not 0.0 <= float(self.suitability_lower_quantile) <= 1.0:
            raise ValueError("suitability_lower_quantile must lie in [0,1].")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AuditThresholds:
    """Pass/warn boundaries for p95 diagnostics unless noted otherwise."""

    model_order_tv_pass: float = 0.05
    model_order_tv_warn: float = 0.10
    peak_count_tv_pass: float = 0.05
    peak_count_tv_warn: float = 0.10
    l2_directional_pass: float = 0.05
    l2_directional_warn: float = 0.10
    w1_directional_pass: float = 0.15
    w1_directional_warn: float = 0.30
    w2_directional_pass: float = 0.20
    w2_directional_warn: float = 0.40
    l2_energy_pass: float = 0.01
    l2_energy_warn: float = 0.03
    w1_energy_pass: float = 0.05
    w1_energy_warn: float = 0.15
    w2_energy_pass: float = 0.05
    w2_energy_warn: float = 0.15
    normalized_derived_pass: float = 0.05
    normalized_derived_warn: float = 0.10
    relative_chi_square_pass: float = 0.10
    relative_chi_square_warn: float = 0.25
    maximum_chi_square_pass: float = 4.0
    maximum_chi_square_warn: float = 8.0
    minimum_w2_spread_pass: float = 0.05
    minimum_w2_spread_warn: float = 0.02
    minimum_w2_floor_ratio_pass: float = 5.0
    minimum_w2_floor_ratio_warn: float = 2.0
    minimum_reference_unique_fraction_pass: float = 0.999
    minimum_reference_unique_fraction_warn: float = 0.995
    maximum_reference_multiplicity_pass: float = 3.0
    maximum_reference_multiplicity_warn: float = 3.0
    maximum_reference_k3_zero_radius_fraction_pass: float = 0.0
    maximum_reference_k3_zero_radius_fraction_warn: float = 0.0

    def __post_init__(self) -> None:
        values = dataclasses.asdict(self)
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for prefix in (
            "model_order_tv",
            "peak_count_tv",
            "l2_directional",
            "w1_directional",
            "w2_directional",
            "l2_energy",
            "w1_energy",
            "w2_energy",
            "normalized_derived",
            "relative_chi_square",
            "maximum_chi_square",
            "maximum_reference_multiplicity",
            "maximum_reference_k3_zero_radius_fraction",
        ):
            if values[f"{prefix}_pass"] > values[f"{prefix}_warn"]:
                raise ValueError(f"{prefix}_pass must not exceed {prefix}_warn.")
        for prefix in (
            "minimum_w2_spread",
            "minimum_w2_floor_ratio",
            "minimum_reference_unique_fraction",
        ):
            if values[f"{prefix}_pass"] < values[f"{prefix}_warn"]:
                raise ValueError(f"{prefix}_pass must not be below {prefix}_warn.")

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def _required(payload: Mapping[str, Any], key: str, artifact_name: str) -> Any:
    if key not in payload:
        raise ValueError(f"{artifact_name} is missing required field {key!r}.")
    return payload[key]


def _tensor_equal(first: torch.Tensor, second: torch.Tensor, *, atol: float) -> bool:
    if tuple(first.shape) != tuple(second.shape):
        return False
    if first.is_floating_point() or second.is_floating_point():
        return bool(
            torch.allclose(
                first.detach().cpu().to(torch.float64),
                second.detach().cpu().to(torch.float64),
                rtol=0.0,
                atol=atol,
            )
        )
    return bool(torch.equal(first.detach().cpu(), second.detach().cpu()))


def _validate_grid(
    grid: Mapping[str, Any],
    expected_grid: Mapping[str, Any],
    *,
    artifact_name: str,
) -> None:
    if grid.get("specification_hash") != expected_grid["specification_hash"]:
        raise ValueError(f"{artifact_name}.grid specification hash differs.")
    for field in _GRID_TENSOR_FIELDS:
        value = _required(grid, field, f"{artifact_name}.grid")
        expected = expected_grid[field]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{artifact_name}.grid[{field!r}] must be a tensor.")
        tolerance = 1.0e-12 if expected.is_floating_point() else 0.0
        if not _tensor_equal(value, expected, atol=tolerance):
            raise ValueError(f"{artifact_name}.grid[{field!r}] differs from the spec.")


def _validate_histograms(samples: torch.Tensor, *, name: str) -> None:
    if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
        raise ValueError(f"{name} must contain finite floating-point samples.")
    if bool((samples < 0).any()):
        raise ValueError(f"{name} contains negative histogram mass.")
    masses = samples.sum(dim=-1)
    if not bool(torch.allclose(masses, torch.ones_like(masses), atol=1e-5, rtol=1e-5)):
        raise ValueError(f"{name} contains a non-unit histogram.")


def validate_reference_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
) -> ContinuousNMRSpec:
    """Strictly validate identities and reference-run independence metadata."""

    if len(artifacts) < 2:
        raise ValueError("at least two independent reference artifacts are required.")
    first_specification = _required(artifacts[0], "specification", "reference 0")
    if not isinstance(first_specification, dict):
        raise ValueError("reference 0 specification must be a dictionary.")
    spec = ContinuousNMRSpec.from_dict(first_specification)
    expected_grid = build_nmr_grid(spec).as_artifact()
    first_settings = _required(artifacts[0], "smc_settings", "reference 0")
    first_ids = _required(artifacts[0], "condition_ids", "reference 0")
    first_measurements = _required(artifacts[0], "measurements", "reference 0")
    first_truth = artifacts[0].get("truth_histograms")
    first_truth_counts = artifacts[0].get("truth_component_counts")
    if not isinstance(first_ids, torch.Tensor) or not isinstance(
        first_measurements, torch.Tensor
    ):
        raise ValueError("condition_ids and measurements must be tensors.")
    if not isinstance(first_truth, torch.Tensor) or not isinstance(
        first_truth_counts, torch.Tensor
    ):
        raise ValueError("truth_histograms and truth_component_counts must be tensors.")
    if first_ids.ndim != 1 or first_ids.numel() < 1:
        raise ValueError("condition_ids must be a nonempty vector.")
    groups = int(first_ids.shape[0])
    if tuple(first_measurements.shape) != (groups, spec.num_measurements):
        raise ValueError("reference 0 measurements have the wrong shape.")
    if tuple(first_truth.shape) != (groups, spec.num_spectrum_bins):
        raise ValueError("reference 0 truth_histograms have the wrong shape.")
    if tuple(first_truth_counts.shape) != (groups,):
        raise ValueError("reference 0 truth_component_counts have the wrong shape.")

    run_ids: list[int] = []
    for index, artifact in enumerate(artifacts):
        name = f"reference {index}"
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{name} must be a dictionary-like artifact.")
        if artifact.get("protocol_id") != spec.protocol_id:
            raise ValueError(f"{name} uses a different protocol_id.")
        if artifact.get("specification_hash") != spec.specification_hash:
            raise ValueError(f"{name} uses a different specification hash.")
        local_specification = _required(artifact, "specification", name)
        if local_specification != spec.to_dict():
            raise ValueError(f"{name} embeds a different specification.")
        if _required(artifact, "smc_settings", name) != first_settings:
            raise ValueError("reference artifacts use different SMC settings.")
        run_id = _required(artifact, "reference_run_id", name)
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 0:
            raise ValueError(f"{name} reference_run_id must be nonnegative.")
        run_ids.append(run_id)
        master_seed = _required(artifact, "reference_master_seed", name)
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or master_seed < 0
        ):
            raise ValueError(f"{name} reference_master_seed must be nonnegative.")
        grid = _required(artifact, "grid", name)
        if not isinstance(grid, Mapping):
            raise ValueError(f"{name}.grid must be a dictionary.")
        _validate_grid(grid, expected_grid, artifact_name=name)

        condition_ids = _required(artifact, "condition_ids", name)
        measurements = _required(artifact, "measurements", name)
        truth = artifact.get("truth_histograms")
        truth_counts = artifact.get("truth_component_counts")
        histograms = _required(artifact, "reference_histograms", name)
        probabilities = _required(
            artifact, "posterior_component_count_probability", name
        )
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                condition_ids,
                measurements,
                histograms,
                probabilities,
            )
        ):
            raise ValueError(f"{name} identity and posterior fields must be tensors.")
        if condition_ids.ndim != 1 or int(condition_ids.shape[0]) != groups:
            raise ValueError(f"{name} condition_ids have the wrong shape.")
        if not _tensor_equal(condition_ids, first_ids, atol=0.0):
            raise ValueError(f"{name} condition IDs differ from reference 0.")
        if tuple(measurements.shape) != (groups, spec.num_measurements):
            raise ValueError(f"{name} measurements have the wrong shape.")
        if not _tensor_equal(measurements, first_measurements, atol=1.0e-12):
            raise ValueError(f"{name} measurements differ from reference 0.")
        if not isinstance(truth, torch.Tensor) or not isinstance(
            truth_counts, torch.Tensor
        ):
            raise ValueError(f"{name} truth fields must be tensors.")
        if tuple(truth.shape) != (groups, spec.num_spectrum_bins):
            raise ValueError(f"{name} truth_histograms have the wrong shape.")
        if not _tensor_equal(truth, first_truth, atol=1.0e-12):
            raise ValueError(f"{name} truth histograms differ from reference 0.")
        if tuple(truth_counts.shape) != (groups,):
            raise ValueError(f"{name} truth_component_counts have the wrong shape.")
        if not _tensor_equal(truth_counts, first_truth_counts, atol=0.0):
            raise ValueError(
                f"{name} truth component counts differ from reference 0."
            )
        if histograms.ndim != 3 or tuple(histograms.shape[::2]) != (
            groups,
            spec.num_spectrum_bins,
        ):
            raise ValueError(f"{name} reference_histograms must have shape [G,S,N].")
        if int(histograms.shape[1]) < sample_count:
            raise ValueError(
                f"{name} contains fewer than the requested {sample_count} samples."
            )
        if int(histograms.shape[1]) <= 3:
            raise ValueError(
                f"{name} must contain more than three samples for the k=3 "
                "duplicate audit."
            )
        _validate_histograms(histograms, name=f"{name}.reference_histograms")
        expected_orders = (
            spec.component_count_maximum - spec.component_count_minimum + 1
        )
        if tuple(probabilities.shape) != (groups, expected_orders):
            raise ValueError(
                f"{name} posterior_component_count_probability has the wrong shape."
            )
        if not probabilities.is_floating_point() or not bool(
            torch.isfinite(probabilities).all()
        ):
            raise ValueError(f"{name} model-order probabilities must be finite.")
        if bool((probabilities < 0).any()) or not bool(
            torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities.sum(dim=-1)),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise ValueError(
                f"{name} model-order probabilities must be simplex-valued."
            )

    if len(set(run_ids)) != len(run_ids):
        raise ValueError("reference artifacts must have distinct run IDs.")
    if int(torch.unique(first_ids).numel()) != int(first_ids.numel()):
        raise ValueError("condition_ids must be unique.")
    return spec


def _subsample(
    samples: torch.Tensor,
    *,
    sample_count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed % (2**63 - 1))
    indices = torch.randperm(int(samples.shape[0]), generator=generator)[:sample_count]
    return samples.detach().cpu().index_select(0, indices)


def _summary(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "maximum": float(tensor.max().item()),
    }


def _derived_scale(name: str, spec: ContinuousNMRSpec) -> float:
    if name.startswith("rate_"):
        return 1.0
    if name == "mean_log_rate":
        return math.log(spec.rate_maximum / spec.rate_minimum)
    if name == "mean_rate":
        return spec.rate_maximum - spec.rate_minimum
    if name == "effective_peak_count":
        return float(spec.component_count_maximum)
    raise ValueError(
        f"No audit normalization is defined for derived quantity {name!r}."
    )


def _derived_comparison(
    first: torch.Tensor,
    second: torch.Tensor,
    log_rate_edges: torch.Tensor,
    spec: ContinuousNMRSpec,
) -> tuple[dict[str, Any], float]:
    # The shared derived-quantity helper normalizes float64 inputs in place.
    # An audit must never modify its source bank while comparing run pairs.
    first_quantities = derived_quantities(first.clone(), log_rate_edges)
    second_quantities = derived_quantities(second.clone(), log_rate_edges)
    first_summary = summarize_derived_quantities(first_quantities)
    second_summary = summarize_derived_quantities(second_quantities)
    result: dict[str, Any] = {}
    maximum_normalized = 0.0
    for name in first_summary:
        absolute = {
            field: abs(first_summary[name][field] - second_summary[name][field])
            for field in ("mean", "q05", "median", "q95")
        }
        normalized = {
            field: value / _derived_scale(name, spec)
            for field, value in absolute.items()
        }
        if name != "effective_peak_count":
            maximum_normalized = max(maximum_normalized, *normalized.values())
        result[name] = {
            "first": first_summary[name],
            "second": second_summary[name],
            "absolute_difference": absolute,
            "normalized_absolute_difference": normalized,
            "used_in_continuous_derived_gate": name != "effective_peak_count",
        }
        if name == "effective_peak_count":
            result[name]["probability_mass_comparison"] = _peak_count_comparison(
                first_quantities[name], second_quantities[name]
            )
    return result, maximum_normalized


def _peak_count_comparison(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    """Compare morphology probabilities, not discontinuous integer quantiles."""

    counts = []
    for values in (first, second):
        work = values.detach().cpu().to(torch.float64).reshape(-1)
        if (
            work.numel() == 0
            or not bool(torch.isfinite(work).all())
            or bool((work < 0).any())
            or not bool(torch.equal(work, work.round()))
        ):
            raise ValueError("effective peak counts must be nonnegative integers.")
        counts.append(work.to(torch.int64))
    support_size = max(int(count.max().item()) for count in counts) + 1
    probabilities = [
        torch.bincount(count, minlength=support_size).to(torch.float64) / count.numel()
        for count in counts
    ]
    difference = probabilities[0] - probabilities[1]
    return {
        "support": list(range(support_size)),
        "first_probability": probabilities[0].tolist(),
        "second_probability": probabilities[1].tolist(),
        "first_sample_count": counts[0].numel(),
        "second_sample_count": counts[1].numel(),
        "total_variation": float(0.5 * difference.abs().sum().item()),
        "maximum_cdf_difference": float(difference.cumsum(0).abs().max().item()),
    }


def _same_bank_split_diagnostics(
    samples: torch.Tensor,
    log_rate_edges: torch.Tensor,
    spec: ContinuousNMRSpec,
    *,
    seed: int,
) -> dict[str, Any]:
    """Report finite-bank variability without treating it as SMC convergence."""

    shuffled = _subsample(samples, sample_count=int(samples.shape[0]), seed=seed)
    midpoint = int(shuffled.shape[0]) // 2
    derived, maximum_normalized = _derived_comparison(
        shuffled[:midpoint], shuffled[midpoint:], log_rate_edges, spec
    )
    return {
        "seed": seed,
        "first_sample_count": midpoint,
        "second_sample_count": int(shuffled.shape[0]) - midpoint,
        "maximum_normalized_continuous_derived_difference": maximum_normalized,
        "peak_count_comparison": derived["effective_peak_count"][
            "probability_mass_comparison"
        ],
        "used_as_acceptance_gate": False,
        "interpretation": (
            "Random disjoint halves of one stored bank illustrate finite-sample "
            "variability only. Correlated SMC particles and shared missed regions "
            "prevent interpreting this as an independent convergence check or "
            "using it to relax cross-run thresholds."
        ),
    }


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    """Summarize a nonempty vector without silently flattening conditions."""

    work = values.detach().cpu().to(torch.float64).reshape(-1)
    if work.numel() < 1 or not bool(torch.isfinite(work).all()):
        raise ValueError("diagnostic values must be nonempty and finite.")
    quantiles = torch.quantile(
        work, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
    )
    return {
        "mean": float(work.mean().item()),
        "standard_deviation": float(work.std(unbiased=False).item()),
        "q05": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "q95": float(quantiles[2].item()),
    }


def _within_posterior_spread(
    samples: torch.Tensor,
    log_rate_edges: torch.Tensor,
    settings: AuditSettings,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    off_diagonal = ~torch.eye(
        int(samples.shape[0]), dtype=torch.bool, device=samples.device
    )
    for metric in GROUND_METRICS:
        distances = pairwise_distances(
            samples,
            samples,
            metric=metric,
            log_rate_edges=log_rate_edges,
            num_quantiles=settings.w2_num_quantiles,
            chunk_size=settings.distance_chunk_size,
        )
        result[metric] = _distribution_summary(distances[off_diagonal])
    return result


def _paired_round_trip_errors(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    log_rate_edges: torch.Tensor,
    settings: AuditSettings,
) -> dict[str, dict[str, float]]:
    """Compute paired errors via chunked calls to the public distance API."""

    result: dict[str, dict[str, float]] = {}
    for metric in GROUND_METRICS:
        pieces: list[torch.Tensor] = []
        for start in range(0, int(original.shape[0]), settings.distance_chunk_size):
            stop = min(start + settings.distance_chunk_size, int(original.shape[0]))
            local = pairwise_distances(
                original[start:stop],
                reconstructed[start:stop],
                metric=metric,
                log_rate_edges=log_rate_edges,
                num_quantiles=settings.w2_num_quantiles,
                chunk_size=settings.distance_chunk_size,
            )
            pieces.append(local.diagonal())
        result[metric] = _distribution_summary(torch.cat(pieces))
    return result


def _posterior_derived_spread(
    samples: torch.Tensor,
    log_rate_edges: torch.Tensor,
    spec: ContinuousNMRSpec,
) -> dict[str, dict[str, float]]:
    quantities = derived_quantities(samples.clone(), log_rate_edges)
    result: dict[str, dict[str, float]] = {}
    for name, values in quantities.items():
        summary = _distribution_summary(values)
        scale = _derived_scale(name, spec)
        result[name] = {
            **summary,
            "q90_width": summary["q95"] - summary["q05"],
            "normalization_scale": scale,
            "normalized_standard_deviation": summary["standard_deviation"] / scale,
            "normalized_q90_width": (summary["q95"] - summary["q05"]) / scale,
        }
    return result


def _benchmark_suitability_records(
    artifacts: Sequence[Mapping[str, Any]],
    spec: ContinuousNMRSpec,
    settings: AuditSettings,
) -> list[dict[str, Any]]:
    log_rate_edges = artifacts[0]["grid"]["log_rate_edges"].to(torch.float64)
    normalized_edges = make_normalized_log_rate_edges(log_rate_edges)
    result: list[dict[str, Any]] = []
    for artifact_index, artifact in enumerate(artifacts):
        for condition_index, condition_id in enumerate(
            artifact["condition_ids"].tolist()
        ):
            subset_seed = (
                settings.seed
                + 500_000_003
                + artifact_index * 10_000_019
                + condition_index * 1_000_003
            )
            full_samples = artifact["reference_histograms"][condition_index].to(
                torch.float64
            )
            duplicate_diagnostics = exact_duplicate_diagnostics(
                full_samples,
                nearest_k=3,
            )
            samples = _subsample(
                full_samples,
                sample_count=settings.sample_count,
                seed=subset_seed,
            ).to(torch.float64)
            quantiles = histogram_to_quantiles(
                samples,
                normalized_edges,
                num_quantiles=settings.representation_quantiles,
            )
            reconstructed = quantiles_to_histograms(quantiles, normalized_edges)
            within_spread = _within_posterior_spread(samples, log_rate_edges, settings)
            round_trip = _paired_round_trip_errors(
                samples, reconstructed, log_rate_edges, settings
            )
            w2_spread = within_spread["w2"]["mean"]
            w2_floor = round_trip["w2"]["mean"]
            floor_denominator = max(w2_floor, 1.0e-12)
            result.append(
                {
                    "artifact_index": artifact_index,
                    "run_id": artifact["reference_run_id"],
                    "condition_index": condition_index,
                    "condition_id": condition_id,
                    "full_bank_sample_count": int(full_samples.shape[0]),
                    "distance_sample_count": int(samples.shape[0]),
                    "subset_seed": subset_seed,
                    "reference_duplicate_diagnostics": duplicate_diagnostics,
                    "within_posterior_offdiagonal_spread": within_spread,
                    "quantile_round_trip_error": round_trip,
                    "w2_spread_to_round_trip_floor_ratio": (
                        w2_spread / floor_denominator
                    ),
                    "round_trip_floor_denominator_clamped": w2_floor < 1.0e-12,
                    "derived_quantity_posterior_spread": _posterior_derived_spread(
                        full_samples, log_rate_edges, spec
                    ),
                    "same_bank_split_diagnostics": _same_bank_split_diagnostics(
                        full_samples,
                        log_rate_edges,
                        spec,
                        seed=subset_seed + 7907,
                    ),
                }
            )
    return result


def _benchmark_suitability_aggregate(
    records: Sequence[dict[str, Any]],
    settings: AuditSettings,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "lower_quantile_probability": settings.suitability_lower_quantile,
        "within_posterior_offdiagonal_spread": {},
        "quantile_round_trip_error": {},
        "derived_quantity_posterior_spread": {},
        "reference_duplicate_diagnostics": {},
    }
    duplicate_records = [
        record["reference_duplicate_diagnostics"] for record in records
    ]
    unique_fractions = [
        float(record["unique_sample_fraction"]) for record in duplicate_records
    ]
    maximum_multiplicities = [
        float(record["maximum_multiplicity"]) for record in duplicate_records
    ]
    zero_radius_fractions = [
        float(record["kth_neighbour_zero_radius_fraction"])
        for record in duplicate_records
    ]
    result["reference_duplicate_diagnostics"] = {
        "unique_sample_fraction": _summary(unique_fractions),
        "minimum_unique_sample_fraction": min(unique_fractions),
        "maximum_multiplicity": _summary(maximum_multiplicities),
        "maximum_maximum_multiplicity": max(maximum_multiplicities),
        "k3_zero_radius_fraction": _summary(zero_radius_fractions),
        "maximum_k3_zero_radius_fraction": max(zero_radius_fractions),
    }
    for metric in GROUND_METRICS:
        result["within_posterior_offdiagonal_spread"][metric] = {}
        result["quantile_round_trip_error"][metric] = {}
        for field in ("mean", "standard_deviation", "q05", "median", "q95"):
            result["within_posterior_offdiagonal_spread"][metric][field] = _summary(
                [
                    record["within_posterior_offdiagonal_spread"][metric][field]
                    for record in records
                ]
            )
            result["quantile_round_trip_error"][metric][field] = _summary(
                [
                    record["quantile_round_trip_error"][metric][field]
                    for record in records
                ]
            )
    ratios = torch.tensor(
        [record["w2_spread_to_round_trip_floor_ratio"] for record in records],
        dtype=torch.float64,
    )
    w2_spreads = torch.tensor(
        [
            record["within_posterior_offdiagonal_spread"]["w2"]["mean"]
            for record in records
        ],
        dtype=torch.float64,
    )
    result["w2_spread_mean_lower_quantile"] = float(
        torch.quantile(w2_spreads, settings.suitability_lower_quantile).item()
    )
    result["w2_spread_to_round_trip_floor_ratio"] = {
        **_summary(ratios.tolist()),
        "lower_quantile": float(
            torch.quantile(ratios, settings.suitability_lower_quantile).item()
        ),
        "denominator_clamped_record_count": sum(
            bool(record["round_trip_floor_denominator_clamped"]) for record in records
        ),
    }
    derived_names = records[0]["derived_quantity_posterior_spread"].keys()
    for name in derived_names:
        result["derived_quantity_posterior_spread"][name] = {}
        for field in (
            "standard_deviation",
            "q90_width",
            "normalized_standard_deviation",
            "normalized_q90_width",
        ):
            result["derived_quantity_posterior_spread"][name][field] = _summary(
                [
                    record["derived_quantity_posterior_spread"][name][field]
                    for record in records
                ]
            )
    return result


def _flatten_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model_order_tv": _summary([record["model_order_tv"] for record in records]),
        "peak_count_tv": _summary([record["peak_count_tv"] for record in records]),
        "peak_count_maximum_cdf_difference": _summary(
            [record["peak_count_maximum_cdf_difference"] for record in records]
        ),
        "maximum_normalized_derived_difference": _summary(
            [record["maximum_normalized_derived_difference"] for record in records]
        ),
        "relative_forward_chi_square_difference": _summary(
            [record["relative_forward_chi_square_difference"] for record in records]
        ),
        "maximum_forward_chi_square": _summary(
            [record["maximum_forward_chi_square"] for record in records]
        ),
        "ground_metrics": {},
        "derived_quantities": {},
        "forward_diagnostics": {},
    }
    for ground_metric in GROUND_METRICS:
        metrics["ground_metrics"][ground_metric] = {}
        for field in ("validity_distance", "coverage_distance", "energy_distance"):
            metrics["ground_metrics"][ground_metric][field] = _summary(
                [record["ground_metrics"][ground_metric][field] for record in records]
            )
    derived_names = records[0]["derived_quantities"].keys()
    for name in derived_names:
        metrics["derived_quantities"][name] = {}
        for field in ("mean", "q05", "median", "q95"):
            metrics["derived_quantities"][name][field] = _summary(
                [
                    record["derived_quantities"][name][
                        "normalized_absolute_difference"
                    ][field]
                    for record in records
                ]
            )
    forward_names = records[0]["forward_diagnostics"]["absolute_difference"].keys()
    for name in forward_names:
        metrics["forward_diagnostics"][name] = _summary(
            [
                record["forward_diagnostics"]["absolute_difference"][name]
                for record in records
            ]
        )
    return metrics


def _status(value: float, pass_threshold: float, warn_threshold: float) -> str:
    if value <= pass_threshold:
        return "pass"
    if value <= warn_threshold:
        return "warn"
    return "fail"


def _minimum_status(value: float, pass_threshold: float, warn_threshold: float) -> str:
    if value >= pass_threshold:
        return "pass"
    if value >= warn_threshold:
        return "warn"
    return "fail"


def _suitability_checks(
    aggregate: Mapping[str, Any], thresholds: AuditThresholds
) -> tuple[list[dict[str, Any]], str]:
    minimum_candidates = (
        (
            "within_posterior_w2_spread_lower_quantile",
            aggregate["w2_spread_mean_lower_quantile"],
            thresholds.minimum_w2_spread_pass,
            thresholds.minimum_w2_spread_warn,
        ),
        (
            "w2_spread_to_round_trip_floor_ratio_lower_quantile",
            aggregate["w2_spread_to_round_trip_floor_ratio"]["lower_quantile"],
            thresholds.minimum_w2_floor_ratio_pass,
            thresholds.minimum_w2_floor_ratio_warn,
        ),
        (
            "reference_unique_fraction_minimum",
            aggregate["reference_duplicate_diagnostics"][
                "minimum_unique_sample_fraction"
            ],
            thresholds.minimum_reference_unique_fraction_pass,
            thresholds.minimum_reference_unique_fraction_warn,
        ),
    )
    maximum_candidates = (
        (
            "reference_maximum_multiplicity",
            aggregate["reference_duplicate_diagnostics"][
                "maximum_maximum_multiplicity"
            ],
            thresholds.maximum_reference_multiplicity_pass,
            thresholds.maximum_reference_multiplicity_warn,
        ),
        (
            "reference_k3_zero_radius_fraction_maximum",
            aggregate["reference_duplicate_diagnostics"][
                "maximum_k3_zero_radius_fraction"
            ],
            thresholds.maximum_reference_k3_zero_radius_fraction_pass,
            thresholds.maximum_reference_k3_zero_radius_fraction_warn,
        ),
    )
    checks = [
        {
            "name": name,
            "value": value,
            "pass_threshold": pass_threshold,
            "warn_threshold": warn_threshold,
            "comparison": "greater_than_or_equal",
            "status": _minimum_status(value, pass_threshold, warn_threshold),
        }
        for name, value, pass_threshold, warn_threshold in minimum_candidates
    ]
    checks.extend(
        {
            "name": name,
            "value": value,
            "pass_threshold": pass_threshold,
            "warn_threshold": warn_threshold,
            "comparison": "less_than_or_equal",
            "status": _status(value, pass_threshold, warn_threshold),
        }
        for name, value, pass_threshold, warn_threshold in maximum_candidates
    )
    ranks = {"pass": 0, "warn": 1, "fail": 2}
    overall = max((check["status"] for check in checks), key=ranks.__getitem__)
    return checks, overall


def _acceptance_checks(
    aggregate: Mapping[str, Any], thresholds: AuditThresholds
) -> tuple[list[dict[str, Any]], str]:
    candidates = [
        (
            "model_order_tv_p95",
            aggregate["model_order_tv"]["p95"],
            thresholds.model_order_tv_pass,
            thresholds.model_order_tv_warn,
        ),
        (
            "peak_count_tv_p95",
            aggregate["peak_count_tv"]["p95"],
            thresholds.peak_count_tv_pass,
            thresholds.peak_count_tv_warn,
        ),
        (
            "normalized_derived_difference_p95",
            aggregate["maximum_normalized_derived_difference"]["p95"],
            thresholds.normalized_derived_pass,
            thresholds.normalized_derived_warn,
        ),
        (
            "relative_forward_chi_square_difference_p95",
            aggregate["relative_forward_chi_square_difference"]["p95"],
            thresholds.relative_chi_square_pass,
            thresholds.relative_chi_square_warn,
        ),
        (
            "maximum_forward_chi_square",
            aggregate["maximum_forward_chi_square"]["maximum"],
            thresholds.maximum_chi_square_pass,
            thresholds.maximum_chi_square_warn,
        ),
    ]
    for metric in GROUND_METRICS:
        directional = max(
            aggregate["ground_metrics"][metric]["validity_distance"]["p95"],
            aggregate["ground_metrics"][metric]["coverage_distance"]["p95"],
        )
        candidates.extend(
            (
                (
                    f"{metric}_directional_distance_p95",
                    directional,
                    getattr(thresholds, f"{metric}_directional_pass"),
                    getattr(thresholds, f"{metric}_directional_warn"),
                ),
                (
                    f"{metric}_energy_distance_p95",
                    aggregate["ground_metrics"][metric]["energy_distance"]["p95"],
                    getattr(thresholds, f"{metric}_energy_pass"),
                    getattr(thresholds, f"{metric}_energy_warn"),
                ),
            )
        )
    checks = [
        {
            "name": name,
            "value": value,
            "pass_threshold": pass_threshold,
            "warn_threshold": warn_threshold,
            "status": _status(value, pass_threshold, warn_threshold),
        }
        for name, value, pass_threshold, warn_threshold in candidates
    ]
    ranks = {"pass": 0, "warn": 1, "fail": 2}
    overall = max((check["status"] for check in checks), key=ranks.__getitem__)
    return checks, overall


def audit_reference_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    settings: AuditSettings,
    thresholds: AuditThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = validate_reference_artifacts(artifacts, sample_count=settings.sample_count)
    condition_ids = artifacts[0]["condition_ids"]
    log_rate_edges = artifacts[0]["grid"]["log_rate_edges"].to(torch.float64)
    kernel = artifacts[0]["grid"]["kernel"].to(torch.float64)
    records: list[dict[str, Any]] = []
    for pair_index, (first_index, second_index) in enumerate(
        itertools.combinations(range(len(artifacts)), 2)
    ):
        first_artifact = artifacts[first_index]
        second_artifact = artifacts[second_index]
        for condition_index, condition_id in enumerate(condition_ids.tolist()):
            seed_base = (
                settings.seed + pair_index * 10_000_019 + condition_index * 1_000_003
            )
            first_full = first_artifact["reference_histograms"][condition_index].to(
                torch.float64
            )
            second_full = second_artifact["reference_histograms"][condition_index].to(
                torch.float64
            )
            first = _subsample(
                first_full,
                sample_count=settings.sample_count,
                seed=seed_base,
            ).to(torch.float64)
            second = _subsample(
                second_full,
                sample_count=settings.sample_count,
                seed=seed_base + 499,
            ).to(torch.float64)
            ground_metrics = {
                metric: compute_sample_set_metrics(
                    first,
                    second,
                    metric=metric,
                    log_rate_edges=log_rate_edges,
                    num_quantiles=settings.w2_num_quantiles,
                    chunk_size=settings.distance_chunk_size,
                    include_prdc=False,
                )
                for metric in GROUND_METRICS
            }
            first_p_j = first_artifact["posterior_component_count_probability"][
                condition_index
            ].to(torch.float64)
            second_p_j = second_artifact["posterior_component_count_probability"][
                condition_index
            ].to(torch.float64)
            model_order_tv = float(0.5 * (first_p_j - second_p_j).abs().sum().item())
            derived, maximum_normalized = _derived_comparison(
                first_full, second_full, log_rate_edges, spec
            )
            peak_comparison = derived["effective_peak_count"][
                "probability_mass_comparison"
            ]
            measurement = first_artifact["measurements"][condition_index].to(
                torch.float64
            )
            first_forward = forward_residual_diagnostics(
                first_full,
                measurement,
                kernel,
                noise_standard_deviation=spec.noise_standard_deviation,
            )
            second_forward = forward_residual_diagnostics(
                second_full,
                measurement,
                kernel,
                noise_standard_deviation=spec.noise_standard_deviation,
            )
            forward_absolute = {
                name: abs(first_forward[name] - second_forward[name])
                for name in first_forward
            }
            first_chi = first_forward["forward_reduced_chi_square_mean"]
            second_chi = second_forward["forward_reduced_chi_square_mean"]
            relative_chi = abs(first_chi - second_chi) / max(
                1.0e-12, 0.5 * (abs(first_chi) + abs(second_chi))
            )
            records.append(
                {
                    "first_artifact_index": first_index,
                    "second_artifact_index": second_index,
                    "first_run_id": first_artifact["reference_run_id"],
                    "second_run_id": second_artifact["reference_run_id"],
                    "condition_index": condition_index,
                    "condition_id": condition_id,
                    "first_full_bank_sample_count": int(first_full.shape[0]),
                    "second_full_bank_sample_count": int(second_full.shape[0]),
                    "distance_sample_count": settings.sample_count,
                    "model_order_tv": model_order_tv,
                    "peak_count_tv": peak_comparison["total_variation"],
                    "peak_count_maximum_cdf_difference": peak_comparison[
                        "maximum_cdf_difference"
                    ],
                    "first_model_order_probability": first_p_j.tolist(),
                    "second_model_order_probability": second_p_j.tolist(),
                    "ground_metrics": ground_metrics,
                    "derived_quantities": derived,
                    "maximum_normalized_derived_difference": maximum_normalized,
                    "forward_diagnostics": {
                        "first": first_forward,
                        "second": second_forward,
                        "absolute_difference": forward_absolute,
                    },
                    "relative_forward_chi_square_difference": relative_chi,
                    "maximum_forward_chi_square": max(first_chi, second_chi),
                }
            )
    aggregate = _flatten_records(records)
    agreement_checks, agreement_status = _acceptance_checks(aggregate, thresholds)
    suitability_records = _benchmark_suitability_records(artifacts, spec, settings)
    suitability_aggregate = _benchmark_suitability_aggregate(
        suitability_records, settings
    )
    suitability_checks, suitability_status = _suitability_checks(
        suitability_aggregate, thresholds
    )
    ranks = {"pass": 0, "warn": 1, "fail": 2}
    overall_status = max((agreement_status, suitability_status), key=ranks.__getitem__)
    identity = {
        "protocol_id": spec.protocol_id,
        "specification_hash": spec.specification_hash,
        "reference_run_ids": [artifact["reference_run_id"] for artifact in artifacts],
        "reference_master_seeds": [
            artifact["reference_master_seed"] for artifact in artifacts
        ],
        "condition_count": int(condition_ids.numel()),
        "condition_ids": condition_ids.tolist(),
        "pair_count": len(artifacts) * (len(artifacts) - 1) // 2,
    }
    raw = {
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        **identity,
        "settings": settings.to_dict(),
        "thresholds": thresholds.to_dict(),
        "agreement_records": records,
        "benchmark_suitability_records": suitability_records,
    }
    summary = {
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        **identity,
        "settings": settings.to_dict(),
        "thresholds": thresholds.to_dict(),
        "agreement": {
            "record_count": len(records),
            "aggregate": aggregate,
            "checks": agreement_checks,
            "status": agreement_status,
            "interpretation": (
                "Continuous derived summaries, forward diagnostics, and effective "
                "peak-count probabilities use every stored sample per condition. "
                "The sample_count setting only limits distance calculations. "
                "Integer peak-count quantile differences are descriptive; peak-count "
                "PMF total variation has its own independent acceptance gate."
            ),
        },
        "agreement_status": agreement_status,
        "benchmark_suitability": {
            "record_count": len(suitability_records),
            "quantile_representation": (
                quantile_wasserstein_representation_signature()
            ),
            "representation_quantiles": settings.representation_quantiles,
            "aggregate": suitability_aggregate,
            "checks": suitability_checks,
            "status": suitability_status,
            "interpretation": (
                "Suitability is independent of cross-run convergence. It requires "
                "the within-posterior W2 spread to be nontrivial and sufficiently "
                "larger than the L=representation_quantiles hard-bin round-trip "
                "floor, and rejects exact-resampling duplicates that would collapse "
                "strict-open k=3 PRDC radii."
            ),
        },
        "suitability_status": suitability_status,
        "overall_status": overall_status,
        "interpretation": (
            "Agreement thresholds test numerical repeatability across independent "
            "SMC runs. Suitability thresholds separately test whether the benchmark "
            "posterior is non-degenerate above the QWFM representation floor and is "
            "not contaminated by duplicate-heavy resampling artifacts. Neither set "
            "of thresholds is a learned-model performance target."
        ),
    }
    return raw, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit agreement among two or more independent SMC references."
    )
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument(
        "--num_cpu_threads",
        type=int,
        default=2,
        help="PyTorch CPU threads; small posterior matrices rarely benefit from large thread pools.",
    )
    parser.add_argument("--w2_num_quantiles", type=int, default=2048)
    parser.add_argument("--distance_chunk_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026090305)
    parser.add_argument(
        "--representation_quantiles",
        type=int,
        default=128,
        help="QWFM midpoint-quantile count used for the representation floor.",
    )
    parser.add_argument(
        "--suitability_lower_quantile",
        type=float,
        default=0.25,
        help="Lower run-condition quantile used by non-degeneracy gates.",
    )
    for field in dataclasses.fields(AuditThresholds):
        parser.add_argument(
            f"--{field.name}",
            type=float,
            default=field.default,
            help=f"Default: {field.default}.",
        )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail_on_rejection",
        action="store_true",
        help="Exit with status 2 when the overall audit status is fail.",
    )
    args = parser.parse_args(argv)
    if len(args.references) < 2:
        parser.error("--references requires at least two artifacts.")
    if args.num_cpu_threads < 1:
        parser.error("--num_cpu_threads must be a positive integer.")
    try:
        AuditSettings(
            sample_count=args.sample_count,
            w2_num_quantiles=args.w2_num_quantiles,
            distance_chunk_size=args.distance_chunk_size,
            seed=args.seed,
            representation_quantiles=args.representation_quantiles,
            suitability_lower_quantile=args.suitability_lower_quantile,
        )
        AuditThresholds(
            **{
                field.name: getattr(args, field.name)
                for field in dataclasses.fields(AuditThresholds)
            }
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.num_cpu_threads)
    output_directory = Path(args.output_dir).resolve()
    raw_path = output_directory / RAW_FILENAME
    summary_path = output_directory / SUMMARY_FILENAME
    if not args.overwrite and (raw_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Audit output already exists in {output_directory}; use --overwrite."
        )
    artifacts = [load_torch_artifact(path) for path in args.references]
    settings = AuditSettings(
        sample_count=args.sample_count,
        w2_num_quantiles=args.w2_num_quantiles,
        distance_chunk_size=args.distance_chunk_size,
        seed=args.seed,
        representation_quantiles=args.representation_quantiles,
        suitability_lower_quantile=args.suitability_lower_quantile,
    )
    thresholds = AuditThresholds(
        **{
            field.name: getattr(args, field.name)
            for field in dataclasses.fields(AuditThresholds)
        }
    )
    raw, summary = audit_reference_artifacts(
        artifacts, settings=settings, thresholds=thresholds
    )
    reference_paths = [str(Path(path).resolve()) for path in args.references]
    raw["reference_paths"] = reference_paths
    summary["reference_paths"] = reference_paths
    atomic_write_json(raw_path, raw)
    atomic_write_json(summary_path, summary)
    print(f"reference agreement: {summary['agreement_status'].upper()}", flush=True)
    for check in summary["agreement"]["checks"]:
        print(
            f"  {check['status'].upper():4s} {check['name']}: "
            f"{check['value']:.6g} (pass <= {check['pass_threshold']:.6g}, "
            f"warn <= {check['warn_threshold']:.6g})",
            flush=True,
        )
    print(
        f"benchmark suitability: {summary['suitability_status'].upper()}",
        flush=True,
    )
    for check in summary["benchmark_suitability"]["checks"]:
        relation = ">=" if check["comparison"] == "greater_than_or_equal" else "<="
        print(
            f"  {check['status'].upper():4s} {check['name']}: "
            f"{check['value']:.6g} (pass {relation} "
            f"{check['pass_threshold']:.6g}, warn {relation} "
            f"{check['warn_threshold']:.6g})",
            flush=True,
        )
    print(f"overall audit: {summary['overall_status'].upper()}", flush=True)
    print(f"saved: {raw_path}", flush=True)
    print(f"saved: {summary_path}", flush=True)
    if args.fail_on_rejection and summary["overall_status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_PROTOCOL_VERSION",
    "AuditSettings",
    "AuditThresholds",
    "audit_reference_artifacts",
    "validate_reference_artifacts",
]
