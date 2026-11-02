"""Ground distances between one-dimensional NMR relaxation spectra.

The metric code is intentionally independent of both the learned models and
the reference-posterior implementation.  A spectrum is represented by the
probability mass in bins that are uniform on the natural-log-rate coordinate.
Consequently, ``w1`` and ``w2`` below compare distributions on log rate rather
than treating the bin-mass vector as an unstructured image.

All public pairwise functions accept either two sample sets, ``[S, N]`` and
``[T, N]``, or two equally grouped sets, ``[G, S, N]`` and ``[G, T, N]``.
Pair blocks are evaluated separately so the temporary tensor is bounded by
``chunk_size ** 2 * N`` (or ``* num_quantiles`` for W2).  Returned distances
use float64 for reproducible offline evaluation even when saved samples are
float32.
"""

from __future__ import annotations

import math
from typing import Final

import torch


GROUND_METRICS: Final[tuple[str, ...]] = ("l2", "w1", "w2")
DISTANCE_PROTOCOL_VERSION: Final[str] = "nmr_log_rate_ground_distances_v1"
_METRIC_ALIASES: Final[dict[str, str]] = {
    "l2": "l2",
    "histogram_l2": "l2",
    "w1": "w1",
    "log_w1": "w1",
    "w2": "w2",
    "log_w2": "w2",
}


def canonical_metric_name(metric: str) -> str:
    """Resolve a ground-metric spelling to ``l2``, ``w1``, or ``w2``."""

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")
    try:
        return _METRIC_ALIASES[metric.lower()]
    except KeyError as error:
        supported = ", ".join(GROUND_METRICS)
        raise ValueError(
            f"unsupported ground metric {metric!r}; use {supported}."
        ) from error


def _validated_chunk_size(
    chunk_size: int | None, left_count: int, right_count: int
) -> int:
    if chunk_size is None:
        return max(left_count, right_count)
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer or None.")
    return chunk_size


def _validate_histogram_tensor(histograms: torch.Tensor, name: str) -> None:
    if not isinstance(histograms, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if histograms.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape [S,N] or [G,S,N].")
    if histograms.shape[-2] < 1 or histograms.shape[-1] < 2:
        raise ValueError(f"{name} must contain samples and at least two bins.")
    if not histograms.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(histograms).all()):
        raise ValueError(f"{name} contains a non-finite value.")
    if bool((histograms < 0).any()):
        raise ValueError(f"{name} contains negative histogram mass.")
    masses = histograms.sum(dim=-1)
    if bool((masses <= 0).any()):
        raise ValueError(f"every row of {name} must have positive mass.")
    if not bool(
        torch.allclose(
            masses,
            torch.ones_like(masses),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    ):
        raise ValueError(f"every row of {name} must have unit mass.")


def _validate_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    _validate_histogram_tensor(left, "left")
    _validate_histogram_tensor(right, "right")
    if left.ndim != right.ndim:
        raise ValueError("left and right must have the same number of dimensions.")
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("left and right must have the same number of bins.")
    if left.ndim == 3 and left.shape[0] != right.shape[0]:
        raise ValueError(
            "grouped left and right inputs must have the same group count."
        )
    if left.device != right.device:
        raise ValueError("left and right must be on the same device.")


def validate_log_rate_edges(
    log_rate_edges: torch.Tensor,
    *,
    num_bins: int | None = None,
) -> torch.Tensor:
    """Validate natural-log-rate bin edges without changing their values."""

    if not isinstance(log_rate_edges, torch.Tensor):
        raise TypeError("log_rate_edges must be a torch.Tensor.")
    if log_rate_edges.ndim != 1 or log_rate_edges.numel() < 3:
        raise ValueError("log_rate_edges must contain at least three values.")
    if not log_rate_edges.is_floating_point():
        raise TypeError("log_rate_edges must have a floating-point dtype.")
    if not bool(torch.isfinite(log_rate_edges).all()):
        raise ValueError("log_rate_edges contains a non-finite value.")
    if bool((torch.diff(log_rate_edges) <= 0).any()):
        raise ValueError("log_rate_edges must be strictly increasing.")
    if num_bins is not None and log_rate_edges.numel() != num_bins + 1:
        raise ValueError("log_rate_edges must contain exactly N + 1 values.")
    return log_rate_edges


def _normalized_work_histograms(histograms: torch.Tensor) -> torch.Tensor:
    work = histograms.to(dtype=torch.float64)
    return work / work.sum(dim=-1, keepdim=True)


def histograms_to_quantiles(
    histograms: torch.Tensor,
    log_rate_edges: torch.Tensor,
    *,
    num_quantiles: int = 2048,
    row_chunk_size: int | None = None,
) -> torch.Tensor:
    """Evaluate inverse CDFs on midpoint probabilities.

    Histogram density is taken to be constant within each log-rate bin.  The
    result has shape ``[..., num_quantiles]`` and float64 dtype.  Midpoint
    quadrature avoids evaluating the ambiguous inverse CDF at probabilities 0
    and 1.
    """

    _validate_histogram_tensor(histograms, "histograms")
    edges = validate_log_rate_edges(log_rate_edges, num_bins=int(histograms.shape[-1]))
    if edges.device != histograms.device:
        raise ValueError("histograms and log_rate_edges must be on the same device.")
    if (
        isinstance(num_quantiles, bool)
        or not isinstance(num_quantiles, int)
        or num_quantiles < 1
    ):
        raise ValueError("num_quantiles must be a positive integer.")

    leading_shape = histograms.shape[:-1]
    flat = _normalized_work_histograms(histograms).reshape(-1, histograms.shape[-1])
    rows = int(flat.shape[0])
    resolved_chunk = _validated_chunk_size(row_chunk_size, rows, rows)
    probabilities = (
        torch.arange(num_quantiles, device=flat.device, dtype=torch.float64) + 0.5
    ) / float(num_quantiles)
    work_edges = edges.to(device=flat.device, dtype=torch.float64)
    result = torch.empty((rows, num_quantiles), device=flat.device, dtype=torch.float64)

    for start in range(0, rows, resolved_chunk):
        stop = min(start + resolved_chunk, rows)
        mass = flat[start:stop]
        cdf = torch.cat(
            (
                torch.zeros((stop - start, 1), device=flat.device, dtype=torch.float64),
                mass.cumsum(dim=-1),
            ),
            dim=-1,
        )
        cdf[:, -1] = 1.0
        query = probabilities[None, :].expand(stop - start, -1).contiguous()
        indices = torch.searchsorted(cdf.contiguous(), query, right=True) - 1
        indices.clamp_(0, mass.shape[-1] - 1)
        left_cdf = cdf.gather(1, indices)
        selected_mass = mass.gather(1, indices)
        if bool((selected_mass <= 0).any()):
            raise RuntimeError("inverse-CDF lookup selected an empty histogram bin.")
        fraction = ((query - left_cdf) / selected_mass).clamp_(0.0, 1.0)
        left_edge = work_edges[indices]
        right_edge = work_edges[indices + 1]
        result[start:stop] = left_edge + fraction * (right_edge - left_edge)

    return result.reshape(*leading_shape, num_quantiles)


def _pairwise_vector_l2(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_size: int | None,
) -> torch.Tensor:
    left_count = int(left.shape[0])
    right_count = int(right.shape[0])
    resolved_chunk = _validated_chunk_size(chunk_size, left_count, right_count)
    result = torch.empty(
        (left_count, right_count), device=left.device, dtype=torch.float64
    )
    for left_start in range(0, left_count, resolved_chunk):
        left_stop = min(left_start + resolved_chunk, left_count)
        for right_start in range(0, right_count, resolved_chunk):
            right_stop = min(right_start + resolved_chunk, right_count)
            result[left_start:left_stop, right_start:right_stop] = torch.cdist(
                left[left_start:left_stop],
                right[right_start:right_stop],
                p=2.0,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
    return result


def _pairwise_l2_2d(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_size: int | None,
) -> torch.Tensor:
    return _pairwise_vector_l2(
        _normalized_work_histograms(left),
        _normalized_work_histograms(right),
        chunk_size=chunk_size,
    )


def _pairwise_w1_2d(
    left: torch.Tensor,
    right: torch.Tensor,
    log_rate_edges: torch.Tensor,
    *,
    chunk_size: int | None,
) -> torch.Tensor:
    left_work = _normalized_work_histograms(left)
    right_work = _normalized_work_histograms(right)
    left_count, right_count = int(left.shape[0]), int(right.shape[0])
    resolved_chunk = _validated_chunk_size(chunk_size, left_count, right_count)
    widths = torch.diff(log_rate_edges.to(device=left.device, dtype=torch.float64))
    left_cdf = torch.cat(
        (torch.zeros_like(left_work[:, :1]), left_work.cumsum(dim=-1)), dim=-1
    )
    right_cdf = torch.cat(
        (torch.zeros_like(right_work[:, :1]), right_work.cumsum(dim=-1)), dim=-1
    )
    left_cdf[:, -1] = 1.0
    right_cdf[:, -1] = 1.0
    result = torch.empty(
        (left_count, right_count), device=left.device, dtype=torch.float64
    )

    for left_start in range(0, left_count, resolved_chunk):
        left_stop = min(left_start + resolved_chunk, left_count)
        for right_start in range(0, right_count, resolved_chunk):
            right_stop = min(right_start + resolved_chunk, right_count)
            boundary_difference = (
                left_cdf[left_start:left_stop, None, :]
                - right_cdf[None, right_start:right_stop, :]
            )
            left_boundary = boundary_difference[..., :-1]
            right_boundary = boundary_difference[..., 1:]
            same_sign_area = 0.5 * widths * (left_boundary.abs() + right_boundary.abs())
            crossing_area = (
                0.5
                * widths
                * (left_boundary.square() + right_boundary.square())
                / (left_boundary.abs() + right_boundary.abs()).clamp_min(1.0e-15)
            )
            bin_area = torch.where(
                left_boundary * right_boundary >= 0.0,
                same_sign_area,
                crossing_area,
            )
            result[left_start:left_stop, right_start:right_stop] = bin_area.sum(-1)
    return result


def _pairwise_w2_2d(
    left: torch.Tensor,
    right: torch.Tensor,
    log_rate_edges: torch.Tensor,
    *,
    num_quantiles: int,
    chunk_size: int | None,
) -> torch.Tensor:
    left_quantiles = histograms_to_quantiles(
        left,
        log_rate_edges,
        num_quantiles=num_quantiles,
        row_chunk_size=chunk_size,
    )
    right_quantiles = histograms_to_quantiles(
        right,
        log_rate_edges,
        num_quantiles=num_quantiles,
        row_chunk_size=chunk_size,
    )
    return _pairwise_vector_l2(
        left_quantiles,
        right_quantiles,
        chunk_size=chunk_size,
    ) / math.sqrt(float(num_quantiles))


def pairwise_distances(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    metric: str,
    log_rate_edges: torch.Tensor | None = None,
    num_quantiles: int = 2048,
    chunk_size: int | None = 64,
) -> torch.Tensor:
    """Return all within-condition ground distances.

    ``l2`` compares normalized bin-mass vectors.  ``w1`` is the exact
    piecewise-uniform one-dimensional Wasserstein-1 distance.  ``w2`` is a
    midpoint inverse-CDF approximation with ``num_quantiles`` points.
    """

    _validate_pair(left, right)
    canonical = canonical_metric_name(metric)
    edges: torch.Tensor | None = None
    if canonical in ("w1", "w2"):
        if log_rate_edges is None:
            raise ValueError(f"log_rate_edges is required for metric={canonical!r}.")
        edges = validate_log_rate_edges(log_rate_edges, num_bins=int(left.shape[-1]))
        if edges.device != left.device:
            raise ValueError(
                "histograms and log_rate_edges must be on the same device."
            )

    if left.ndim == 3:
        grouped = [
            pairwise_distances(
                left[group_index],
                right[group_index],
                metric=canonical,
                log_rate_edges=edges,
                num_quantiles=num_quantiles,
                chunk_size=chunk_size,
            )
            for group_index in range(int(left.shape[0]))
        ]
        return torch.stack(grouped, dim=0)

    if canonical == "l2":
        return _pairwise_l2_2d(left, right, chunk_size=chunk_size)
    if canonical == "w1":
        assert edges is not None
        return _pairwise_w1_2d(left, right, edges, chunk_size=chunk_size)
    assert edges is not None
    return _pairwise_w2_2d(
        left,
        right,
        edges,
        num_quantiles=num_quantiles,
        chunk_size=chunk_size,
    )


__all__ = [
    "DISTANCE_PROTOCOL_VERSION",
    "GROUND_METRICS",
    "canonical_metric_name",
    "histograms_to_quantiles",
    "pairwise_distances",
    "validate_log_rate_edges",
]
