"""Condition-wise distribution metrics for posterior reconstruction samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..utils.duplicate_utils import exact_duplicate_diagnostics
from .distances import (
    DISTANCE_PROTOCOL_VERSION,
    GROUND_METRICS,
    canonical_metric_name,
    pairwise_distances,
)
from .prdc import (
    PRDC_PROTOCOL_VERSION,
    compute_prdc_from_distances,
    validate_finite_metrics,
)


METRIC_PROTOCOL_VERSION = "continuous_posterior_metrics_v1"


def _validate_cross_distance_matrix(distances: torch.Tensor) -> None:
    if not isinstance(distances, torch.Tensor) or distances.ndim != 2:
        raise ValueError("reference_generated_distances must be a matrix.")
    if distances.shape[0] < 1 or distances.shape[1] < 1:
        raise ValueError("distance matrices must contain both sample sets.")
    if not distances.is_floating_point():
        raise TypeError("distance matrices must have a floating-point dtype.")
    if not bool(torch.isfinite(distances).all()) or bool((distances < -1.0e-12).any()):
        raise ValueError("distance matrices must be finite and nonnegative.")


def directional_distances_from_cross(
    reference_generated_distances: torch.Tensor,
) -> dict[str, float]:
    """Compute threshold-free nearest-neighbour validity and coverage.

    Rows of the input are reference samples and columns are generated samples.
    ``validity_distance`` therefore averages generated-to-reference minima,
    whereas ``coverage_distance`` averages reference-to-generated minima.
    Both are lower-is-better.
    """

    _validate_cross_distance_matrix(reference_generated_distances)
    validity = reference_generated_distances.min(dim=0).values.mean()
    coverage = reference_generated_distances.min(dim=1).values.mean()
    return {
        "validity_distance": float(validity.item()),
        "coverage_distance": float(coverage.item()),
    }


def energy_distance_from_matrices(
    reference_reference_distances: torch.Tensor,
    generated_generated_distances: torch.Tensor,
    reference_generated_distances: torch.Tensor,
) -> float:
    """Return the empirical V-statistic energy distance.

    Diagonal within-set distances are included.  This biased/V-statistic form
    is nonnegative for the three negative-type ground metrics used here and is
    exactly zero for identical empirical sample sets.  Tiny negative roundoff
    is clipped to zero.
    """

    _validate_cross_distance_matrix(reference_reference_distances)
    _validate_cross_distance_matrix(generated_generated_distances)
    _validate_cross_distance_matrix(reference_generated_distances)
    reference_count = int(reference_reference_distances.shape[0])
    generated_count = int(generated_generated_distances.shape[0])
    if tuple(reference_reference_distances.shape) != (
        reference_count,
        reference_count,
    ):
        raise ValueError("reference_reference_distances must be square.")
    if tuple(generated_generated_distances.shape) != (
        generated_count,
        generated_count,
    ):
        raise ValueError("generated_generated_distances must be square.")
    if tuple(reference_generated_distances.shape) != (
        reference_count,
        generated_count,
    ):
        raise ValueError("cross-distance dimensions do not match within-set matrices.")

    value = (
        2.0 * reference_generated_distances.to(torch.float64).mean()
        - reference_reference_distances.to(torch.float64).mean()
        - generated_generated_distances.to(torch.float64).mean()
    )
    return float(value.clamp_min(0.0).item())


def sample_set_diagnostics(samples: torch.Tensor) -> dict[str, float | int]:
    """Report simplex validity and exact empirical sample multiplicity."""

    if not isinstance(samples, torch.Tensor) or samples.ndim != 2:
        raise ValueError("samples must have shape [S,N].")
    if samples.shape[0] < 1 or samples.shape[1] < 2:
        raise ValueError("samples must contain at least one row and two bins.")
    if not samples.is_floating_point():
        raise TypeError("samples must have a floating-point dtype.")
    finite_by_sample = torch.isfinite(samples).all(dim=-1)
    finite_values = torch.where(
        torch.isfinite(samples), samples, torch.zeros_like(samples)
    )
    negative_by_sample = (finite_values < 0).any(dim=-1)
    mass = finite_values.sum(dim=-1)
    mass_error = (mass - 1.0).abs()
    simplex_valid = finite_by_sample & ~negative_by_sample & (mass_error <= 1.0e-5)
    finite_rows = samples[finite_by_sample]
    if int(finite_rows.shape[0]) > 3:
        duplicate_diagnostics = exact_duplicate_diagnostics(
            finite_rows,
            nearest_k=3,
        )
    else:
        _, counts = torch.unique(finite_rows.detach().cpu(), dim=0, return_counts=True)
        unique_count = int(counts.numel())
        duplicate_diagnostics = {
            "unique_sample_count": unique_count,
            "unique_sample_fraction": unique_count / float(samples.shape[0]),
            "maximum_multiplicity": 0 if unique_count == 0 else int(counts.max()),
            "kth_neighbour_zero_radius_fraction": 0.0,
        }
    return {
        "sample_count": int(samples.shape[0]),
        "unique_sample_count": duplicate_diagnostics["unique_sample_count"],
        "unique_sample_fraction": (
            duplicate_diagnostics["unique_sample_count"] / float(samples.shape[0])
        ),
        "maximum_multiplicity": duplicate_diagnostics["maximum_multiplicity"],
        "k3_zero_radius_fraction": duplicate_diagnostics[
            "kth_neighbour_zero_radius_fraction"
        ],
        "finite_sample_fraction": float(
            finite_by_sample.to(torch.float64).mean().item()
        ),
        "negative_sample_fraction": float(
            negative_by_sample.to(torch.float64).mean().item()
        ),
        "simplex_valid_fraction": float(simplex_valid.to(torch.float64).mean().item()),
        "mass_error_mean": float(mass_error.to(torch.float64).mean().item()),
        "mass_error_maximum": float(mass_error.max().item()),
    }


def compute_sample_set_metrics(
    reference: torch.Tensor,
    generated: torch.Tensor,
    *,
    metric: str,
    log_rate_edges: torch.Tensor | None = None,
    nearest_k: int = 3,
    num_quantiles: int = 2048,
    chunk_size: int | None = 64,
    include_prdc: bool = True,
    strict_prdc: bool = True,
) -> dict[str, float]:
    """Evaluate two posterior sample sets under one NMR ground metric."""

    if reference.ndim != 2 or generated.ndim != 2:
        raise ValueError("reference and generated must both have shape [S,N].")
    cross = pairwise_distances(
        reference,
        generated,
        metric=metric,
        log_rate_edges=log_rate_edges,
        num_quantiles=num_quantiles,
        chunk_size=chunk_size,
    )
    reference_reference = pairwise_distances(
        reference,
        reference,
        metric=metric,
        log_rate_edges=log_rate_edges,
        num_quantiles=num_quantiles,
        chunk_size=chunk_size,
    )
    generated_generated = pairwise_distances(
        generated,
        generated,
        metric=metric,
        log_rate_edges=log_rate_edges,
        num_quantiles=num_quantiles,
        chunk_size=chunk_size,
    )
    result = directional_distances_from_cross(cross)
    result["energy_distance"] = energy_distance_from_matrices(
        reference_reference, generated_generated, cross
    )
    if include_prdc:
        result.update(
            compute_prdc_from_distances(
                reference_reference,
                generated_generated,
                cross,
                nearest_k=nearest_k,
                strict=strict_prdc,
            )
        )
    validate_finite_metrics(result)
    return result


def _aggregate_scalars(
    records: Sequence[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    keys = tuple(records[0].keys())
    if any(tuple(record.keys()) != keys for record in records):
        raise ValueError("all records must have the same scalar fields.")
    aggregate: dict[str, dict[str, float]] = {}
    for key in keys:
        values = torch.tensor([record[key] for record in records], dtype=torch.float64)
        aggregate[key] = {
            "mean": float(values.mean().item()),
            "standard_deviation": float(values.std(unbiased=False).item()),
            "minimum": float(values.min().item()),
            "maximum": float(values.max().item()),
        }
    return aggregate


def evaluate_grouped_samples(
    references: torch.Tensor,
    generated: torch.Tensor,
    log_rate_edges: torch.Tensor,
    *,
    ground_metrics: Sequence[str] = GROUND_METRICS,
    nearest_k: int = 3,
    num_quantiles: int = 2048,
    chunk_size: int | None = 64,
    include_prdc: bool = True,
    strict_prdc: bool = True,
    require_matched_counts: bool = True,
    condition_ids: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Evaluate samples independently for each fixed measurement condition.

    Inputs have shapes ``[G,S,N]`` and ``[G,K,N]``.  No distance is ever
    computed across the leading condition dimension.  Aggregate values weight
    each measurement condition equally, irrespective of its posterior spread.
    """

    if not isinstance(references, torch.Tensor) or references.ndim != 3:
        raise ValueError("references must have shape [G,S,N].")
    if not isinstance(generated, torch.Tensor) or generated.ndim != 3:
        raise ValueError("generated must have shape [G,K,N].")
    groups, reference_count, bins = map(int, references.shape)
    generated_groups, generated_count, generated_bins = map(int, generated.shape)
    if groups < 1 or reference_count < 1 or generated_count < 1:
        raise ValueError("grouped inputs must be nonempty.")
    if (generated_groups, generated_bins) != (groups, bins):
        raise ValueError("references and generated must have matching G and N.")
    if require_matched_counts and reference_count != generated_count:
        raise ValueError(
            "the primary protocol requires matched reference/generated counts; "
            "subsample the reference set first or set require_matched_counts=False."
        )
    canonical_metrics = tuple(canonical_metric_name(name) for name in ground_metrics)
    if not canonical_metrics or len(set(canonical_metrics)) != len(canonical_metrics):
        raise ValueError("ground_metrics must be nonempty and contain no duplicates.")
    if condition_ids is None:
        resolved_ids: list[str | int] = list(range(groups))
    else:
        resolved_ids = list(condition_ids)
        if len(resolved_ids) != groups:
            raise ValueError("condition_ids must have one value per group.")

    per_condition: list[dict[str, Any]] = []
    by_metric: dict[str, list[dict[str, float]]] = {
        name: [] for name in canonical_metrics
    }
    for group_index, condition_id in enumerate(resolved_ids):
        metric_records: dict[str, dict[str, float]] = {}
        for name in canonical_metrics:
            values = compute_sample_set_metrics(
                references[group_index],
                generated[group_index],
                metric=name,
                log_rate_edges=log_rate_edges,
                nearest_k=nearest_k,
                num_quantiles=num_quantiles,
                chunk_size=chunk_size,
                include_prdc=include_prdc,
                strict_prdc=strict_prdc,
            )
            metric_records[name] = values
            by_metric[name].append(values)
        per_condition.append(
            {
                "condition_index": group_index,
                "condition_id": condition_id,
                "reference_diagnostics": sample_set_diagnostics(
                    references[group_index]
                ),
                "generated_diagnostics": sample_set_diagnostics(generated[group_index]),
                "metrics": metric_records,
            }
        )

    return {
        "protocol": {
            "version": METRIC_PROTOCOL_VERSION,
            "distance_protocol_version": DISTANCE_PROTOCOL_VERSION,
            "prdc_protocol_version": PRDC_PROTOCOL_VERSION,
            "condition_wise": True,
            "ground_metrics": list(canonical_metrics),
            "nearest_k": nearest_k,
            "w2_num_quantiles": num_quantiles,
            "prdc_included": include_prdc,
            "prdc_boundary": "strict" if strict_prdc else "closed",
            "reference_sample_count": reference_count,
            "generated_sample_count": generated_count,
        },
        "per_condition": per_condition,
        "aggregate": {
            name: _aggregate_scalars(records) for name, records in by_metric.items()
        },
    }


def deterministic_reference_splits(
    reference: torch.Tensor,
    *,
    sample_count: int,
    repeats: int = 20,
    seed: int = 0,
    include_disjoint_oracle: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Create reproducible matched reference and ref-vs-ref oracle subsets.

    Each repeat uses a new permutation.  When an oracle is requested, its
    samples are disjoint from the primary subset within that repeat.  This
    helper returns tensors rather than indices so callers cannot accidentally
    index a different reference run with the same values.
    """

    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError("reference must have shape [S,N].")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    required = sample_count * (2 if include_disjoint_oracle else 1)
    if int(reference.shape[0]) < required:
        raise ValueError(
            f"reference needs at least {required} rows for the requested split."
        )
    result: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    for repeat in range(repeats):
        generator = torch.Generator(device="cpu")
        generator.manual_seed((seed + 1_000_003 * repeat) % (2**63 - 1))
        indices = torch.randperm(int(reference.shape[0]), generator=generator)[
            :required
        ]
        indices = indices.to(device=reference.device)
        primary = reference.index_select(0, indices[:sample_count])
        oracle = None
        if include_disjoint_oracle:
            oracle = reference.index_select(0, indices[sample_count:required])
        result.append((primary, oracle))
    return result


__all__ = [
    "METRIC_PROTOCOL_VERSION",
    "compute_sample_set_metrics",
    "deterministic_reference_splits",
    "directional_distances_from_cross",
    "energy_distance_from_matrices",
    "evaluate_grouped_samples",
    "sample_set_diagnostics",
]
