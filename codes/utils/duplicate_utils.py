"""Exact duplicate diagnostics for empirical sample clouds."""

from __future__ import annotations

from typing import Any

import torch


def exact_duplicate_diagnostics(
    samples: torch.Tensor,
    *,
    nearest_k: int = 3,
) -> dict[str, Any]:
    """Summarize exact multiplicities and their effect on k-NN radii.

    The diagonal is excluded from a sample's neighbourhood.  Consequently,
    the k-th-neighbour radius is exactly zero iff that sample belongs to an
    exact-duplicate group of size greater than ``nearest_k``.  Computing this
    from multiplicities avoids constructing a quadratic distance matrix.
    """

    if not isinstance(samples, torch.Tensor) or samples.ndim != 2:
        raise ValueError("samples must have shape [S,D].")
    if int(samples.shape[0]) < 1 or int(samples.shape[1]) < 1:
        raise ValueError("samples must be a nonempty matrix.")
    if isinstance(nearest_k, bool) or not isinstance(nearest_k, int) or nearest_k < 1:
        raise ValueError("nearest_k must be a positive integer.")
    if int(samples.shape[0]) <= nearest_k:
        raise ValueError("sample count must exceed nearest_k.")
    if samples.is_floating_point() and not bool(torch.isfinite(samples).all()):
        raise ValueError("samples must be finite.")

    work = samples.detach().to(device="cpu").contiguous()
    _, inverse, counts = torch.unique(
        work,
        dim=0,
        sorted=False,
        return_inverse=True,
        return_counts=True,
    )
    per_sample_multiplicity = counts.index_select(0, inverse)
    zero_radius = per_sample_multiplicity > nearest_k
    sample_count = int(work.shape[0])
    unique_count = int(counts.numel())
    return {
        "sample_count": sample_count,
        "unique_sample_count": unique_count,
        "unique_sample_fraction": unique_count / float(sample_count),
        "maximum_multiplicity": int(counts.max().item()),
        "nearest_k": nearest_k,
        "zero_radius_sample_count": int(zero_radius.sum().item()),
        "kth_neighbour_zero_radius_fraction": float(
            zero_radius.to(torch.float64).mean().item()
        ),
    }
