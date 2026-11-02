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
from ._truth_score import _compute_truth_score_report
from .distances import pairwise_distances


KERNEL_SCORE_PROTOCOL_VERSION = "truth_histogram_gaussian_kernel_score_u_v1"
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
) -> dict[str, Any]:
    """Score [K,N] iid normalized histogram draws against one [N] truth.

    Distances and kernel arithmetic use float64, with chunked distance blocks.
    A declared point forecast has E[k(X,X')]=k(x,x)=1, including K=1.
    A stochastic K=1 forecast cannot estimate the independent-pair term.
    """

    bandwidth = _validated_bandwidth(bandwidth)
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool.")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 2:
        raise ValueError("samples must have shape [K,N].")
    if not isinstance(truth, torch.Tensor) or truth.ndim != 1:
        raise ValueError("truth must have shape [N].")
    if samples.device != truth.device:
        raise ValueError("samples and truth must be on the same device.")
    truth_distances = pairwise_distances(
        samples, truth[None, :], metric="l2", chunk_size=distance_chunk_size
    )
    count = int(samples.shape[0])
    if deterministic and not torch.equal(samples, samples[:1].expand_as(samples)):
        raise ValueError("A deterministic forecast contains different sample rows.")
    result: dict[str, Any] = {
        "sample_count": count,
        "deterministic": deterministic,
        "kernel_bandwidth": bandwidth,
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
        distances = pairwise_distances(
            samples, samples, metric="l2", chunk_size=distance_chunk_size
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
def _calibrated_bandwidth(canonical_specification: str) -> tuple[float, dict[str, Any]]:
    """Cache by complete canonical simulator spec; use only private CPU RNGs."""

    spec = ContinuousNMRSpec.from_dict(json.loads(canonical_specification))
    grid = build_nmr_grid(spec)
    histograms = torch.stack([
        simulate_from_key(
            index, spec=spec, grid=grid,
            master_seed=KERNEL_SCORE_CALIBRATION_SEED,
            split=KERNEL_SCORE_CALIBRATION_SPLIT,
            epoch=KERNEL_SCORE_CALIBRATION_EPOCH,
        )["histogram"]
        for index in range(KERNEL_SCORE_CALIBRATION_COUNT)
    ])
    histogram_sha256 = hashlib.sha256(
        histograms.numpy().astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    distances = pairwise_distances(histograms, histograms, metric="l2", chunk_size=64)
    indices = torch.triu_indices(histograms.shape[0], histograms.shape[0], offset=1)
    distinct_pairs = distances[indices[0], indices[1]]
    # Average the two central order statistics when the pair count is even.
    bandwidth = _validated_bandwidth(float(torch.quantile(distinct_pairs, 0.5).item()))
    provenance: dict[str, Any] = {
        "selection": "median_all_distinct_unordered_pairwise_l2",
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
        "distance": "l2_of_normalized_histogram_bin_masses",
        "median_convention": "quantile_0.5_linear_interpolation",
        "computation_dtype": "float64",
        "calibration_device": "cpu",
        "evaluation_predictions_used": False,
        "evaluation_truths_used": False,
        "posterior_reference_samples_used": False,
    }
    return bandwidth, provenance


def compute_kernel_score_report(
    samples_artifact: Mapping[str, Any],
    truth_artifact: Mapping[str, Any] | None = None,
    *,
    sample_count: int | None = None,
    condition_indices: list[int] | None = None,
    distance_chunk_size: int = 64,
    device: torch.device | str = "cpu",
    bandwidth: float | None = None,
) -> dict[str, Any]:
    """Score cached forecasts against aligned simulator truths, without SMC.

    Identity/shape checks and truth availability match energy-score reports.
    With bandwidth=None, 512 separately keyed prior histograms determine one
    fixed bandwidth per complete simulator spec. An explicit bandwidth must
    already have been fixed independently of evaluation predictions/truths.
    Missing-truth reports do not trigger simulator calibration.
    """

    resolved_bandwidth = None if bandwidth is None else _validated_bandwidth(bandwidth)
    provenance: dict[str, Any] = {
        "selection": (
            "independent_simulator_calibration_pending"
            if bandwidth is None else "explicit_fixed_bandwidth"
        ),
        "calibration_performed": False,
    }

    def prepare_score() -> dict[str, Any]:
        nonlocal resolved_bandwidth
        if resolved_bandwidth is None:
            canonical = json.dumps(
                dict(samples_artifact["specification"]),
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            resolved_bandwidth, cached_provenance = _calibrated_bandwidth(canonical)
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
        score_name="kernel", protocol_version=KERNEL_SCORE_PROTOCOL_VERSION,
        terms=_TERMS,
        score_metadata={
            "kernel": "gaussian_rbf_on_normalized_histogram_bin_masses",
            "kernel_formula": "k(x,z)=exp(-||x-z||_2^2/(2*kernel_bandwidth^2))",
            "kernel_bandwidth": resolved_bandwidth,
            "bandwidth_provenance": provenance,
            "formula": "sum_{i!=j} k(h_i,h_j)/(K*(K-1)) - 2*mean_i k(h_i,h_true)",
            "computation_dtype": "float64",
            "score_interpretation": "proper_kernel_score_not_absolute_conditional_mmd_squared",
        },
        point_iid_assumption=(
            "Explicit point forecast: pairwise kernel is exactly one; repetitions do not add information."
        ),
        score_one=score_one, prepare_score=prepare_score,
    )


__all__ = [
    "KERNEL_SCORE_PROTOCOL_VERSION",
    "compute_kernel_score_report",
    "histogram_gaussian_kernel_score",
]
