"""Truth-based Gaussian kernel scores without posterior-reference samples.

The score is E[k(X,X')] - 2 E[k(X,h_true)], with independent predictive
draws X and X'. Its expected excess over the true forecast equals squared
Gaussian MMD. It is not an estimate of absolute conditional MMD squared.

The default bandwidth uses a separate, fixed simulator calibration split.
Neither predictions, test truths, nor SMC samples select the bandwidth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from functools import lru_cache
from numbers import Real
from typing import Any

import torch

from ..data.grid import build_nmr_grid
from ..data.simulator import simulate_from_key
from ..data.spec import ContinuousNMRSpec
from ._truth_score import (
    _compute_truth_score_report, _distance_metadata, _score_distances, _score_metric,
)


KERNEL_SCORE_PROTOCOL_VERSION = "truth_histogram_gaussian_kernel_score_u_v1"
W2_KERNEL_SCORE_PROTOCOL_VERSION = "truth_histogram_w2_exact_gaussian_kernel_score_u_v1"
KERNEL_SCORE_CALIBRATION_COUNT = 512
KERNEL_SCORE_CALIBRATION_SEED = 2026092801
KERNEL_SCORE_CALIBRATION_SPLIT = "kernel_score_calibration"
KERNEL_SCORE_CALIBRATION_EPOCH = 0
_TERMS = ("mean_kernel_to_truth", "mean_pairwise_kernel", "kernel_score")


def _validated_bandwidth(bandwidth: float) -> float:
    if (
        isinstance(bandwidth, bool) or not isinstance(bandwidth, Real)
        or not math.isfinite(float(bandwidth)) or bandwidth <= 0
    ):
        raise ValueError("bandwidth must be a finite positive real number.")
    return float(bandwidth)


def _gaussian_kernel(distances: torch.Tensor, bandwidth: float) -> torch.Tensor:
    # Divide before squaring: finite extreme bandwidths must not create 0/0
    # on identical samples through underflow of bandwidth**2.
    return torch.exp(-0.5 * (distances / bandwidth).square())


def histogram_gaussian_kernel_score(
    samples: torch.Tensor,
    truth: torch.Tensor,
    *,
    bandwidth: float,
    deterministic: bool = False,
    distance_chunk_size: int = 64,
    metric: str = "l2",
    log_rate_edges: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score [K,N] iid normalized histogram draws against one [N] truth.

    Distances and kernel arithmetic use float64, with chunked distance blocks.
    A declared point forecast has E[k(X,X')]=k(x,x)=1, including K=1.
    A stochastic K=1 forecast cannot estimate the independent-pair term.
    ``metric='w2'`` uses exact W2 between distributions with uniform density
    within the supplied natural-log-rate bins. Its bandwidth must be calibrated
    in W2 units, independently of the evaluated predictions and truths.
    """

    metric = _score_metric(metric)
    bandwidth = _validated_bandwidth(bandwidth)
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
        "kernel_bandwidth": bandwidth,
        "metric": metric,
    }
    if count == 1 and not deterministic:
        return {
            **result,
            "available": False,
            "status": "unavailable",
            "reason": "Stochastic K=1 cannot estimate the independent-pair term; require K>=2.",
        }
    mean_kernel_to_truth = float(_gaussian_kernel(truth_distances, bandwidth).mean().item())
    if deterministic:
        mean_pairwise_kernel = 1.0
    else:
        distances = _score_distances(
            samples, samples, metric=metric, log_rate_edges=log_rate_edges,
            chunk_size=distance_chunk_size,
        )
        kernels = _gaussian_kernel(distances, bandwidth)
        # A repeated but independently drawn value still contributes k(x,x)=1.
        # Exclude only each draw paired with itself, not equal-valued draws.
        kernels.fill_diagonal_(0.0)
        mean_pairwise_kernel = float(kernels.sum().item()) / (count * (count - 1))
    return {
        **result,
        "available": True,
        "status": "available",
        "mean_kernel_to_truth": mean_kernel_to_truth,
        "mean_pairwise_kernel": mean_pairwise_kernel,
        "kernel_score": mean_pairwise_kernel - 2.0 * mean_kernel_to_truth,
    }


@lru_cache(maxsize=16)
def _calibration_histograms(canonical_specification: str) -> torch.Tensor:
    """Replay the existing independent bank once per spec for both metrics."""
    spec = ContinuousNMRSpec.from_dict(json.loads(canonical_specification))
    grid = build_nmr_grid(spec)
    return torch.stack([
        simulate_from_key(
            index, spec=spec, grid=grid,
            master_seed=KERNEL_SCORE_CALIBRATION_SEED,
            split=KERNEL_SCORE_CALIBRATION_SPLIT,
            epoch=KERNEL_SCORE_CALIBRATION_EPOCH,
        )["histogram"]
        for index in range(KERNEL_SCORE_CALIBRATION_COUNT)
    ])


@lru_cache(maxsize=32)
def _calibrated_bandwidth(
    canonical_specification: str, metric: str = "l2",
) -> tuple[float, dict[str, Any]]:
    """Cache by complete simulator spec and metric; use only private CPU RNGs."""

    metric = _score_metric(metric)
    spec = ContinuousNMRSpec.from_dict(json.loads(canonical_specification))
    grid = build_nmr_grid(spec)
    histograms = _calibration_histograms(canonical_specification)
    histogram_sha256 = hashlib.sha256(
        histograms.numpy().astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    distances = _score_distances(
        histograms, histograms, metric=metric,
        log_rate_edges=grid.log_rate_edges, chunk_size=64,
    )
    indices = torch.triu_indices(histograms.shape[0], histograms.shape[0], offset=1)
    distinct_pairs = distances[indices[0], indices[1]]
    # Average the two central order statistics when the pair count is even.
    bandwidth = _validated_bandwidth(float(torch.quantile(distinct_pairs, 0.5).item()))
    provenance: dict[str, Any] = {
        "selection": f"median_all_distinct_unordered_pairwise_{metric}",
        "calibration_performed": True,
        "calibration_split": KERNEL_SCORE_CALIBRATION_SPLIT,
        "calibration_master_seed": KERNEL_SCORE_CALIBRATION_SEED,
        "calibration_epoch": KERNEL_SCORE_CALIBRATION_EPOCH,
        "calibration_count": KERNEL_SCORE_CALIBRATION_COUNT,
        "calibration_sample_ids": [0, KERNEL_SCORE_CALIBRATION_COUNT - 1],
        "calibration_sample_ids_convention": "inclusive_contiguous_range",
        "calibration_pair_count": int(distinct_pairs.numel()),
        "calibration_histogram_sha256": histogram_sha256,
        "calibration_histogram_hash_encoding": "simulator_float32_little_endian_C_order",
        "calibration_specification_hash": spec.specification_hash,
        "distance": (
            "w2_of_piecewise_uniform_natural_log_rate_histograms" if metric == "w2"
            else "l2_of_normalized_histogram_bin_masses"
        ),
        **_distance_metadata(metric),
        "median_convention": "quantile_0.5_linear_interpolation",
        "computation_dtype": "float64",
        "calibration_device": "cpu",
        "evaluation_predictions_used": False,
        "evaluation_truths_used": False,
        "posterior_reference_samples_used": False,
    }
    return bandwidth, provenance


# Preserve the existing explicit cache reset API, including deterministic
# replay tests, while sharing histograms between L2 and W2 during normal use.
_clear_bandwidth_values = _calibrated_bandwidth.cache_clear


def _clear_bandwidth_caches() -> None:
    _clear_bandwidth_values()
    _calibration_histograms.cache_clear()


_calibrated_bandwidth.cache_clear = _clear_bandwidth_caches


def compute_kernel_score_report(
    samples_artifact: Mapping[str, Any],
    truth_artifact: Mapping[str, Any] | None = None,
    *,
    sample_count: int | None = None,
    condition_indices: list[int] | None = None,
    distance_chunk_size: int = 64,
    device: torch.device | str = "cpu",
    bandwidth: float | None = None,
    metric: str = "l2",
    bandwidth_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score cached forecasts against aligned simulator truths, without SMC.

    Identity/shape checks and truth availability match energy-score reports.
    With bandwidth=None, 512 separately keyed prior histograms determine one
    fixed bandwidth per complete simulator spec and metric. Both metrics use
    the same independently keyed calibration histogram bank; W2 bandwidth is
    the median exact W2 distance, never the L2 bandwidth. An explicit bandwidth must
    already have been fixed independently of evaluation predictions/truths.
    Missing-truth reports do not trigger simulator calibration.
    """

    metric = _score_metric(metric)
    resolved_bandwidth = None if bandwidth is None else _validated_bandwidth(bandwidth)
    provenance: dict[str, Any] = {
        "selection": (
            "independent_simulator_calibration_pending"
            if bandwidth is None else "explicit_fixed_bandwidth"
        ),
        "calibration_performed": False,
        "metric": metric,
    }
    if bandwidth_provenance is not None:
        if resolved_bandwidth is None:
            raise ValueError("bandwidth_provenance requires an explicit bandwidth.")
        if not isinstance(bandwidth_provenance, Mapping):
            raise TypeError("bandwidth_provenance must be a mapping.")
        if bandwidth_provenance.get("metric") != metric:
            raise ValueError("Bandwidth provenance metric must match the score metric.")
        provenance.update(json.loads(json.dumps(dict(bandwidth_provenance), allow_nan=False)))
        calibration_hash = provenance.get("calibration_specification_hash")
        if calibration_hash is not None and calibration_hash != samples_artifact.get("specification_hash"):
            raise ValueError("Bandwidth calibration specification hash differs from predictions.")
        for field in ("evaluation_predictions_used", "evaluation_truths_used", "posterior_reference_samples_used"):
            if provenance.get(field, False) is not False:
                raise ValueError("Bandwidth must be selected independently of evaluation and posterior references.")
        expected_protocol = _distance_metadata(metric)["distance_protocol_version"]
        if provenance.get("distance_protocol_version", expected_protocol) != expected_protocol:
            raise ValueError("Bandwidth provenance distance protocol differs from the score metric.")

    def prepare_score() -> dict[str, Any]:
        nonlocal resolved_bandwidth
        if resolved_bandwidth is None:
            canonical = json.dumps(
                dict(samples_artifact["specification"]),
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            resolved_bandwidth, cached_provenance = _calibrated_bandwidth(canonical, metric=metric)
            # Do not expose cached mutable metadata to callers.
            provenance.update(json.loads(json.dumps(cached_provenance)))
        return {
            "kernel_bandwidth": resolved_bandwidth,
            "bandwidth_provenance": provenance,
        }

    def score_one(samples: torch.Tensor, truth: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        if resolved_bandwidth is None:
            raise RuntimeError("Kernel bandwidth must be prepared before scoring.")
        return histogram_gaussian_kernel_score(
            samples, truth, bandwidth=resolved_bandwidth, **kwargs
        )

    return _compute_truth_score_report(
        samples_artifact, truth_artifact,
        sample_count=sample_count, condition_indices=condition_indices,
        distance_chunk_size=distance_chunk_size, device=device,
        score_name="kernel", protocol_version=(
            W2_KERNEL_SCORE_PROTOCOL_VERSION if metric == "w2" else KERNEL_SCORE_PROTOCOL_VERSION
        ),
        terms=_TERMS,
        score_metadata={
            **_distance_metadata(metric),
            "kernel": (
                "gaussian_rbf_on_natural_log_rate_w2" if metric == "w2"
                else "gaussian_rbf_on_normalized_histogram_bin_masses"
            ),
            "kernel_formula": (
                "k(x,z)=exp(-W2(x,z)^2/(2*kernel_bandwidth^2))" if metric == "w2"
                else "k(x,z)=exp(-||x-z||_2^2/(2*kernel_bandwidth^2))"
            ),
            "kernel_bandwidth": resolved_bandwidth,
            "bandwidth_provenance": provenance,
            "formula": "sum_{i!=j} k(h_i,h_j)/(K*(K-1)) - 2*mean_i k(h_i,h_true)",
            "computation_dtype": "float64",
            "score_interpretation": "proper_kernel_score_not_absolute_conditional_mmd_squared",
        },
        point_iid_assumption=(
            "Explicit point forecast: pairwise kernel is exactly one; repetitions do not add information."
        ),
        score_one=score_one, prepare_score=prepare_score, metric=metric,
    )


__all__ = [
    "KERNEL_SCORE_PROTOCOL_VERSION",
    "W2_KERNEL_SCORE_PROTOCOL_VERSION",
    "compute_kernel_score_report",
    "histogram_gaussian_kernel_score",
]
