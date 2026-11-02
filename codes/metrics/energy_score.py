"""Truth-based energy scores, independent of posterior-reference samples.

The U-statistic estimates the population energy score for iid predictive
draws. Its spread term uses distinct pairs and an unsquared distance: Euclidean
histogram-vector distance by default or exact natural-log-rate W2. Conditions, not reference resampling
repeats, are the units of aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._truth_score import (
    _compute_truth_score_report, _distance_metadata, _score_distances, _score_metric,
)


ENERGY_SCORE_PROTOCOL_VERSION = "truth_histogram_l2_energy_score_u_v1"
W2_ENERGY_SCORE_PROTOCOL_VERSION = "truth_histogram_w2_exact_energy_score_u_v1"
_TERMS = ("mean_distance_to_truth", "mean_pairwise_distance", "energy_score")


def histogram_l2_energy_score(
    samples: torch.Tensor,
    truth: torch.Tensor,
    *,
    deterministic: bool = False,
    distance_chunk_size: int = 64,
    metric: str = "l2",
    log_rate_edges: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score one condition using [K,N] draws and its actual [N] truth.

    K=1 is valid only for an explicitly declared point forecast.  A stochastic
    K=1 forecast cannot estimate the independent-pair term and is unavailable.
    Declared deterministic forecasts must contain identical repeated points.
    ``metric='w2'`` requires natural-log-rate edges and integrates the squared
    inverse-CDF difference exactly for uniform density within each bin, then
    takes its square root. The legacy function name retains its L2 default.
    """

    metric = _score_metric(metric)
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool.")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 2:
        raise ValueError("samples must have shape [K,N].")
    if not isinstance(truth, torch.Tensor) or truth.ndim != 1:
        raise ValueError("truth must have shape [N].")
    if samples.device != truth.device:
        raise ValueError("samples and truth must be on the same device.")
    truth_distances = _score_distances(
        samples, truth[None, :], metric=metric, log_rate_edges=log_rate_edges,
        chunk_size=distance_chunk_size,
    )
    count = int(samples.shape[0])
    if deterministic and not torch.equal(samples, samples[:1].expand_as(samples)):
        raise ValueError("A deterministic forecast contains different sample rows.")
    result: dict[str, Any] = {
        "sample_count": count,
        "deterministic": deterministic,
        "metric": metric,
    }
    if count == 1 and not deterministic:
        return {
            **result,
            "available": False,
            "status": "unavailable",
            "reason": "Stochastic K=1 cannot estimate the independent-pair term; require K>=2.",
        }
    distance_to_truth = float(truth_distances.mean().item())
    if deterministic:
        pairwise_distance = 0.0
    else:
        distances = _score_distances(
            samples, samples, metric=metric, log_rate_edges=log_rate_edges,
            chunk_size=distance_chunk_size,
        )
        # Exclude self-pairs explicitly, including any numerical diagonal error.
        distances.fill_diagonal_(0.0)
        pairwise_distance = float(distances.sum().item()) / (count * (count - 1))
    return {
        **result,
        "available": True,
        "status": "available",
        "mean_distance_to_truth": distance_to_truth,
        "mean_pairwise_distance": pairwise_distance,
        "energy_score": distance_to_truth - 0.5 * pairwise_distance,
    }


def compute_energy_score_report(
    samples_artifact: Mapping[str, Any],
    truth_artifact: Mapping[str, Any] | None = None,
    *,
    sample_count: int | None = None,
    condition_indices: list[int] | None = None,
    distance_chunk_size: int = 64,
    device: torch.device | str = "cpu",
    metric: str = "l2",
) -> dict[str, Any]:
    """Score cached predictions against aligned simulator truths, without SMC.

    ``truth_artifact`` may contain reference-style ``truth_histograms``,
    ``condition_ids``, ``measurements`` or fixed-simulator-split ``histogram``,
    ``sample_id``, ``measurement``.  Protocol, specification hash, ordered IDs,
    and observations must match.  A posterior mean/sample is never substituted
    for a truth.  A missing truth artifact returns unavailable.
    """

    metric = _score_metric(metric)
    return _compute_truth_score_report(
        samples_artifact, truth_artifact,
        sample_count=sample_count, condition_indices=condition_indices,
        distance_chunk_size=distance_chunk_size, device=device,
        score_name="energy", protocol_version=(
            W2_ENERGY_SCORE_PROTOCOL_VERSION if metric == "w2" else ENERGY_SCORE_PROTOCOL_VERSION
        ),
        terms=_TERMS,
        score_metadata={
            **_distance_metadata(metric),
            "formula": (
                "mean_i W2(h_i,h_true) - sum_{i!=j} W2(h_i,h_j) / (2*K*(K-1))"
                if metric == "w2" else
                "mean_i ||h_i-h_true||_2 - sum_{i!=j} ||h_i-h_j||_2 / (2*K*(K-1))"
            ),
        },
        point_iid_assumption=(
            "Explicit point forecast: pairwise term is exactly zero; repetitions do not add information."
        ),
        score_one=histogram_l2_energy_score, metric=metric,
    )


__all__ = [
    "ENERGY_SCORE_PROTOCOL_VERSION",
    "W2_ENERGY_SCORE_PROTOCOL_VERSION",
    "compute_energy_score_report",
    "histogram_l2_energy_score",
]
