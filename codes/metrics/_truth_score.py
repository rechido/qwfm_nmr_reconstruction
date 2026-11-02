"""Shared identity checks and condition aggregation for truth-based scores.

Only actual simulator truths are accepted. Posterior-reference ensembles are
never consulted; scores require an independently realized simulator truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..data.grid import build_nmr_grid
from ..data.spec import ContinuousNMRSpec
from .distances import (
    DISTANCE_PROTOCOL_VERSION,
    W2_EXACT_PROTOCOL_VERSION,
    canonical_metric_name,
    pairwise_distances,
    pairwise_w2_exact,
    validate_log_rate_edges,
)


_SYNTHETIC_PROTOCOL = "continuous_variable_gamma_mixture_v1"


def _score_metric(metric: str) -> str:
    metric = canonical_metric_name(metric)
    if metric not in ("l2", "w2"):
        raise ValueError("Truth proper scores support metric='l2' or metric='w2'.")
    return metric


def _score_distances(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    metric: str,
    log_rate_edges: torch.Tensor | None,
    chunk_size: int,
) -> torch.Tensor:
    if _score_metric(metric) == "w2":
        if log_rate_edges is None:
            raise ValueError("metric='w2' requires log_rate_edges.")
        return pairwise_w2_exact(left, right, log_rate_edges, chunk_size=chunk_size)
    return pairwise_distances(left, right, metric="l2", chunk_size=chunk_size)


def _distance_metadata(metric: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "ground_metric": metric,
        "distance_protocol_version": (
            W2_EXACT_PROTOCOL_VERSION if metric == "w2"
            else DISTANCE_PROTOCOL_VERSION
        ),
        "ground_distance": (
            "unsquared_w2_of_piecewise_uniform_natural_log_rate_histograms"
            if metric == "w2" else "unsquared_l2_of_normalized_histogram_bin_masses"
        ),
        "distance_computation": (
            "exact_piecewise_uniform_inverse_cdf_integration" if metric == "w2"
            else "exact_euclidean_norm"
        ),
        "distance_squared": False,
        "quantile_quadrature_used": False,
    }


def _resolved_log_rate_edges(
    samples_artifact: Mapping[str, Any],
    truth_artifact: Mapping[str, Any] | None,
) -> torch.Tensor:
    """Derive the canonical grid and reject conflicting stored grid evidence."""

    spec = ContinuousNMRSpec.from_dict(dict(samples_artifact["specification"]))
    expected = build_nmr_grid(spec).as_artifact()
    for name, artifact in (("samples artifact", samples_artifact), ("truth artifact", truth_artifact)):
        if artifact is None or "grid" not in artifact:
            continue
        grid = artifact["grid"]
        if not isinstance(grid, Mapping):
            raise ValueError(f"{name}.grid must be a mapping.")
        if grid.get("specification_hash") != spec.specification_hash:
            raise ValueError(f"{name}.grid specification hash differs from the specification.")
        for field, canonical in expected.items():
            if not isinstance(canonical, torch.Tensor):
                continue
            value = grid.get(field)
            if not isinstance(value, torch.Tensor) or value.shape != canonical.shape:
                raise ValueError(f"{name}.grid[{field!r}] has an invalid shape or is missing.")
            if field == "log_rate_edges":
                validate_log_rate_edges(value, num_bins=spec.num_spectrum_bins)
            tolerance = 1.0e-12 if canonical.is_floating_point() else 0.0
            if not torch.allclose(
                value.detach().cpu().to(torch.float64), canonical.to(torch.float64),
                rtol=0.0, atol=tolerance,
            ):
                raise ValueError(f"{name}.grid[{field!r}] differs from the specification.")
    return expected["log_rate_edges"]


def _artifact_protocol(artifact: Mapping[str, Any], name: str) -> tuple[str, str]:
    specification = artifact.get("specification")
    if not isinstance(specification, Mapping):
        raise ValueError(f"{name} requires a specification mapping.")
    protocol = artifact.get("protocol_id")
    if protocol != specification.get("protocol_id"):
        raise ValueError(f"{name} protocol_id differs from its specification.")
    serialized = json.dumps(
        dict(specification), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    expected_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if artifact.get("specification_hash") != expected_hash:
        raise ValueError(f"{name} specification hash does not match its specification.")
    return protocol, expected_hash


def _field(artifact: Mapping[str, Any], primary: str, alternate: str) -> Any:
    return artifact[primary] if primary in artifact else artifact.get(alternate)


def _deterministic_declaration(artifact: Mapping[str, Any]) -> tuple[bool, str]:
    if "deterministic" in artifact:
        value = artifact["deterministic"]
        if not isinstance(value, bool):
            raise TypeError("samples artifact deterministic must be a bool.")
        return value, "artifact.deterministic"
    # Existing classical artifacts explicitly declare Dirac replication instead.
    config = artifact.get("sampling_config", {})
    if isinstance(config, Mapping) and "dirac_replication_count" in config:
        count = config["dirac_replication_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("dirac_replication_count must be a positive integer.")
        if count != artifact["samples"].shape[1]:
            raise ValueError("dirac_replication_count differs from the stored sample count.")
        return True, "sampling_config.dirac_replication_count"
    return False, "absent_assumed_stochastic"


def _condition_aggregate(
    records: list[dict[str, Any]], terms: tuple[str, ...]
) -> dict[str, Any]:
    count = len(records)
    result = {}
    for name in terms:
        values = torch.tensor([record[name] for record in records], dtype=torch.float64)
        standard_deviation = float(values.std(correction=1)) if count > 1 else None
        result[name] = {
            "mean": float(values.mean()),
            "standard_deviation": standard_deviation,
            "standard_error": (
                standard_deviation / math.sqrt(count) if count > 1 else None
            ),
            "condition_count": count,
        }
    return result


def _compute_truth_score_report(
    samples_artifact: Mapping[str, Any],
    truth_artifact: Mapping[str, Any] | None,
    *,
    sample_count: int | None,
    condition_indices: list[int] | None,
    distance_chunk_size: int,
    device: torch.device | str,
    score_name: str,
    protocol_version: str,
    terms: tuple[str, ...],
    score_metadata: dict[str, Any],
    point_iid_assumption: str,
    score_one: Callable[..., dict[str, Any]],
    prepare_score: Callable[[], dict[str, Any]] | None = None,
    metric: str = "l2",
) -> dict[str, Any]:
    """Validate aligned artifacts before optional preparation or scoring."""

    metric = _score_metric(metric)
    protocol, specification_hash = _artifact_protocol(samples_artifact, "samples artifact")
    samples = samples_artifact.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 3 or min(samples.shape) < 1:
        raise ValueError("samples must have nonempty shape [G,K,N].")
    groups, stored_count, _ = samples.shape
    if sample_count is None:
        sample_count = int(stored_count)
    if (
        isinstance(sample_count, bool) or not isinstance(sample_count, int)
        or not 1 <= sample_count <= stored_count
    ):
        raise ValueError("sample_count must lie between one and the stored sample count.")
    selected = list(range(groups)) if condition_indices is None else list(condition_indices)
    if (
        not selected or any(isinstance(i, bool) or not isinstance(i, int) for i in selected)
        or len(set(selected)) != len(selected)
        or any(i < 0 or i >= groups for i in selected)
    ):
        raise ValueError("condition_indices must be nonempty, unique, in-range integers.")
    deterministic, declaration = _deterministic_declaration(samples_artifact)
    report: dict[str, Any] = {
        "protocol_version": protocol_version,
        "protocol_id": protocol,
        "specification_hash": specification_hash,
        "direction": "lower_is_better",
        "sample_count": sample_count,
        "condition_count": len(selected),
        "condition_indices": selected,
        "deterministic": deterministic,
        "deterministic_declaration": declaration,
        "iid_assumption": (
            point_iid_assumption
            if deterministic else
            "Off-diagonal estimator assumes iid predictive draws conditional on each measurement; this assumption is not established by this metric."
        ),
        "aggregation_unit": "one_score_per_condition_not_reference_repeat",
        "uncertainty_convention": "Sample SD (ddof=1) and SD/sqrt(G); SE assumes independent test conditions; both null for G=1.",
        "reference_posterior_samples_used": False,
        "per_condition": [],
        "aggregate": {},
        **score_metadata,
    }
    if protocol != _SYNTHETIC_PROTOCOL:
        raise ValueError(f"Truth {score_name} scoring supports the declared synthetic protocol only.")
    log_rate_edges = None
    if metric == "w2":
        log_rate_edges = _resolved_log_rate_edges(samples_artifact, None)
        if samples.shape[-1] != log_rate_edges.numel() - 1:
            raise ValueError("Prediction histogram size differs from the specification.")
        report.update({
            "log_rate_edges": log_rate_edges.tolist(),
            "log_rate_coordinate": "natural_log_rate",
            "histogram_interpretation": "constant_density_within_each_natural_log_rate_bin",
            "grid_source": "specification_validated_against_present_artifact_grids",
        })
    if truth_artifact is None:
        return {
            **report,
            "available": False,
            "status": "unavailable",
            "reason": "No simulator truth artifact is available; posterior references are not a substitute.",
        }
    truth_protocol, truth_hash = _artifact_protocol(truth_artifact, "truth artifact")
    if (protocol, specification_hash) != (truth_protocol, truth_hash):
        raise ValueError("Prediction and truth protocol/specification hashes differ.")
    if metric == "w2":
        log_rate_edges = _resolved_log_rate_edges(samples_artifact, truth_artifact).to(device)
    specification = samples_artifact["specification"]
    if samples.shape[2] != specification.get("num_spectrum_bins"):
        raise ValueError("Prediction histogram size differs from the specification.")
    condition_ids = samples_artifact.get("condition_ids")
    truth_ids = _field(truth_artifact, "condition_ids", "sample_id")
    for ids in (condition_ids, truth_ids):
        if (
            not isinstance(ids, torch.Tensor) or ids.shape != (groups,)
            or ids.dtype not in (torch.int32, torch.int64)
            or torch.unique(ids).numel() != groups
        ):
            raise ValueError("Both artifacts require unique integer condition IDs of shape [G].")
    if not torch.equal(condition_ids.cpu(), truth_ids.cpu()):
        raise ValueError("Prediction and truth condition IDs differ or have different ordering.")
    measurements = samples_artifact.get("measurements")
    truth_measurements = _field(truth_artifact, "measurements", "measurement")
    if (
        not isinstance(measurements, torch.Tensor)
        or not isinstance(truth_measurements, torch.Tensor)
        or measurements.shape != (groups, specification.get("num_measurements"))
        or measurements.shape != truth_measurements.shape
        or not bool(torch.isfinite(measurements).all())
        or not bool(torch.isfinite(truth_measurements).all())
        or not torch.allclose(
            measurements.detach().cpu().to(torch.float64),
            truth_measurements.detach().cpu().to(torch.float64),
            rtol=0.0, atol=1.0e-7,
        )
    ):
        raise ValueError("Prediction and truth measurements differ or have invalid shapes.")
    truth = _field(truth_artifact, "truth_histograms", "histogram")
    if truth is None:
        return {
            **report,
            "available": False,
            "status": "unavailable",
            "reason": "No simulator truth histogram is available; posterior references are not a substitute.",
        }
    if not isinstance(truth, torch.Tensor) or truth.shape != (groups, samples.shape[2]):
        raise ValueError("truth histograms must have shape [G,N] matching predictions.")
    if prepare_score is not None:
        report.update(prepare_score())
    records = []
    for index in selected:
        if deterministic and not torch.equal(
            samples[index], samples[index, :1].expand_as(samples[index])
        ):
            raise ValueError("A deterministic forecast contains different sample rows.")
        score = score_one(
            samples[index, :sample_count].to(device), truth[index].to(device),
            deterministic=deterministic, distance_chunk_size=distance_chunk_size,
            metric=metric, log_rate_edges=log_rate_edges,
        )
        if not score["available"]:
            return {
                **report, "available": False, "status": "unavailable", "reason": score["reason"]
            }
        records.append({
            "condition_index": index,
            "condition_id": int(condition_ids[index]),
            **{name: score[name] for name in terms},
        })
    return {
        **report,
        "available": True,
        "status": "available",
        "truth_source": (
            "simulator_truth_histograms_stored_in_reference_artifact"
            if "truth_histograms" in truth_artifact else "simulator_fixed_split_histogram"
        ),
        "per_condition": records,
        "aggregate": _condition_aggregate(records, terms),
    }
