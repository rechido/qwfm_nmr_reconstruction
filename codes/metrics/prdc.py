"""Precision, recall, density, and coverage for conditional NMR samples.

This is a fresh distance-matrix implementation of the sample-manifold metrics
used in generative-model evaluation.  No natural-image feature extractor is
involved: the caller supplies NMR histogram samples and chooses one of the
physical ground distances from :mod:`codes.metrics.distances`.

The default uses strict open k-nearest-neighbour balls, matching the protocol
fixed for this benchmark.  In particular, a deterministic prediction repeated
K times has zero generated radii and therefore cannot acquire artificial
recall from the equality ``distance == radius == 0``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .distances import pairwise_distances


PRDC_PROTOCOL_VERSION = "conditionwise_nmr_prdc_open_knn_v1"


def _validate_distance_matrix(
    distances: torch.Tensor,
    name: str,
    *,
    square: bool,
) -> None:
    if not isinstance(distances, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if distances.ndim != 2 or distances.shape[0] < 1 or distances.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix.")
    if square and distances.shape[0] != distances.shape[1]:
        raise ValueError(f"{name} must be square.")
    if not distances.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(distances).all()):
        raise ValueError(f"{name} contains a non-finite value.")
    if bool((distances < -1.0e-12).any()):
        raise ValueError(f"{name} contains a negative distance.")


def kth_neighbour_radii(
    within_set_distances: torch.Tensor,
    *,
    nearest_k: int,
) -> torch.Tensor:
    """Return each row's k-th neighbour radius, excluding the diagonal.

    ``nearest_k=1`` means the closest *other* sample.  Duplicate samples are
    retained, so a radius of zero is meaningful evidence of empirical
    collapse rather than something silently repaired by jitter.
    """

    _validate_distance_matrix(within_set_distances, "within_set_distances", square=True)
    count = int(within_set_distances.shape[0])
    if isinstance(nearest_k, bool) or not isinstance(nearest_k, int) or nearest_k < 1:
        raise ValueError("nearest_k must be a positive integer.")
    if count <= nearest_k:
        raise ValueError(
            f"PRDC requires sample_count > nearest_k, got {count} and {nearest_k}."
        )
    without_self = within_set_distances.clone()
    diagonal = torch.arange(count, device=without_self.device)
    without_self[diagonal, diagonal] = torch.inf
    return torch.kthvalue(without_self, k=nearest_k, dim=1).values


def _inside(
    distances: torch.Tensor, radii: torch.Tensor, *, strict: bool
) -> torch.Tensor:
    return distances < radii if strict else distances <= radii


def compute_prdc_from_distances(
    reference_reference_distances: torch.Tensor,
    generated_generated_distances: torch.Tensor,
    reference_generated_distances: torch.Tensor,
    *,
    nearest_k: int = 3,
    strict: bool = True,
) -> dict[str, float]:
    """Compute PRDC from three mutually consistent distance matrices.

    The cross matrix is oriented as ``[reference, generated]``.  Density is
    deliberately not clipped to one: values above one are valid and indicate
    that generated points lie in overlapping reference neighbourhoods.
    """

    _validate_distance_matrix(
        reference_reference_distances,
        "reference_reference_distances",
        square=True,
    )
    _validate_distance_matrix(
        generated_generated_distances,
        "generated_generated_distances",
        square=True,
    )
    _validate_distance_matrix(
        reference_generated_distances,
        "reference_generated_distances",
        square=False,
    )
    reference_count = int(reference_reference_distances.shape[0])
    generated_count = int(generated_generated_distances.shape[0])
    if tuple(reference_generated_distances.shape) != (
        reference_count,
        generated_count,
    ):
        raise ValueError(
            "reference_generated_distances must have shape "
            f"[{reference_count},{generated_count}]."
        )
    devices = {
        reference_reference_distances.device,
        generated_generated_distances.device,
        reference_generated_distances.device,
    }
    if len(devices) != 1:
        raise ValueError("all distance matrices must be on the same device.")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean.")

    reference_radii = kth_neighbour_radii(
        reference_reference_distances, nearest_k=nearest_k
    )
    generated_radii = kth_neighbour_radii(
        generated_generated_distances, nearest_k=nearest_k
    )
    inside_reference = _inside(
        reference_generated_distances,
        reference_radii[:, None],
        strict=strict,
    )
    inside_generated = _inside(
        reference_generated_distances,
        generated_radii[None, :],
        strict=strict,
    )

    precision = inside_reference.any(dim=0).to(torch.float64).mean()
    recall = inside_generated.any(dim=1).to(torch.float64).mean()
    density = inside_reference.sum(dim=0).to(torch.float64).mean() / float(nearest_k)
    nearest_generated = reference_generated_distances.min(dim=1).values
    coverage = (
        _inside(nearest_generated, reference_radii, strict=strict)
        .to(torch.float64)
        .mean()
    )

    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "density": float(density.item()),
        "coverage": float(coverage.item()),
        "reference_radius_mean": float(reference_radii.mean().item()),
        "generated_radius_mean": float(generated_radii.mean().item()),
        "reference_zero_radius_fraction": float(
            (reference_radii == 0).to(torch.float64).mean().item()
        ),
        "generated_zero_radius_fraction": float(
            (generated_radii == 0).to(torch.float64).mean().item()
        ),
    }


def compute_prdc(
    reference: torch.Tensor,
    generated: torch.Tensor,
    *,
    metric: str,
    log_rate_edges: torch.Tensor | None = None,
    nearest_k: int = 3,
    num_quantiles: int = 2048,
    chunk_size: int | None = 64,
    strict: bool = True,
) -> dict[str, float]:
    """Compute NMR-specific PRDC directly from two histogram sample sets."""

    if reference.ndim != 2 or generated.ndim != 2:
        raise ValueError("compute_prdc expects [S,N] reference and generated sets.")
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
    reference_generated = pairwise_distances(
        reference,
        generated,
        metric=metric,
        log_rate_edges=log_rate_edges,
        num_quantiles=num_quantiles,
        chunk_size=chunk_size,
    )
    return compute_prdc_from_distances(
        reference_reference,
        generated_generated,
        reference_generated,
        nearest_k=nearest_k,
        strict=strict,
    )


def validate_finite_metrics(metrics: dict[str, Any]) -> None:
    """Raise when a serialized metric dictionary contains a non-finite scalar."""

    for name, value in metrics.items():
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            raise ValueError(f"metric {name!r} is non-finite: {value!r}.")


__all__ = [
    "PRDC_PROTOCOL_VERSION",
    "compute_prdc",
    "compute_prdc_from_distances",
    "kth_neighbour_radii",
    "validate_finite_metrics",
]
