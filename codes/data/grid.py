from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import torch

from .spec import ContinuousNMRSpec


M32_SUBSET_INDICES = (
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
)


@dataclass(frozen=True)
class NMRGrid:
    log_rate_edges: torch.Tensor
    rate_edges: torch.Tensor
    rate_centers: torch.Tensor
    echo_times_64: torch.Tensor
    echo_subset_indices: torch.Tensor
    echo_times: torch.Tensor
    kernel: torch.Tensor
    specification_hash: str

    def to(self, *args: Any, **kwargs: Any) -> "NMRGrid":
        fields: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            fields[field.name] = value.to(*args, **kwargs) if torch.is_tensor(value) else value
        return NMRGrid(**fields)

    def as_artifact(self) -> dict[str, Any]:
        return {
            "log_rate_edges": self.log_rate_edges.cpu(),
            "rate_edges": self.rate_edges.cpu(),
            "rate_centers": self.rate_centers.cpu(),
            "echo_times_64": self.echo_times_64.cpu(),
            "echo_subset_indices": self.echo_subset_indices.cpu(),
            "echo_times": self.echo_times.cpu(),
            "kernel": self.kernel.cpu(),
            "specification_hash": self.specification_hash,
        }


def build_nmr_grid(
    spec: ContinuousNMRSpec,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> NMRGrid:
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point.")
    log_rate_edges = torch.linspace(
        math.log(spec.rate_minimum),
        math.log(spec.rate_maximum),
        spec.num_spectrum_bins + 1,
        dtype=dtype,
        device=device,
    )
    rate_edges = log_rate_edges.exp()
    rate_centers = torch.exp(0.5 * (log_rate_edges[:-1] + log_rate_edges[1:]))
    echo_times_64 = torch.logspace(
        math.log10(spec.echo_time_minimum),
        math.log10(spec.echo_time_maximum),
        spec.canonical_echo_count,
        dtype=dtype,
        device=device,
    )
    if spec.num_measurements == 64:
        indices = torch.arange(64, dtype=torch.int64, device=device)
    else:
        indices = torch.tensor(M32_SUBSET_INDICES, dtype=torch.int64, device=device)
    echo_times = echo_times_64.index_select(0, indices)
    kernel = torch.exp(-echo_times[:, None] * rate_centers[None, :])
    return NMRGrid(
        log_rate_edges=log_rate_edges,
        rate_edges=rate_edges,
        rate_centers=rate_centers,
        echo_times_64=echo_times_64,
        echo_subset_indices=indices,
        echo_times=echo_times,
        kernel=kernel,
        specification_hash=spec.specification_hash,
    )
