from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import isotonic_regression

from utils.unet_config import unet_sample
from utils.nmr_utils import (
    histogram_to_quantiles,
    make_normalized_log_rate_edges,
    quantiles_to_histograms,
    sample_standard_dirichlet,
)


def _isotonic_projection(values: torch.Tensor) -> torch.Tensor:
    """Exact equal-weight L2 projection onto nondecreasing rows (PAVA).

    Rows that are already valid stay on-device.  Only violating rows use
    SciPy's public O(L) compiled PAVA implementation on CPU; this avoids the
    original Python scalar loop at every Euler step while preserving the exact
    isotonic-regression projection.
    """

    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("values must have shape [rows, quantiles] with L >= 2.")
    violation_mask = (values[:, 1:] < values[:, :-1]).any(dim=1)
    if not bool(violation_mask.any()):
        return values

    violating = (
        values[violation_mask]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    projected_numpy = np.empty_like(violating)
    for row_index, row in enumerate(violating):
        projected_numpy[row_index] = isotonic_regression(row).x
    projected_violating = torch.from_numpy(projected_numpy).to(
        device=values.device,
        dtype=values.dtype,
    )
    if bool(violation_mask.all()):
        return projected_violating
    projected = values.clone()
    projected[violation_mask] = projected_violating
    return projected


class QuantileWassersteinSampler:
    """Explicit Euler sampler in equal-mass quantile coordinates."""

    def __init__(
        self,
        model,
        device: torch.device,
        source_seed: int,
        log_rate_edges: torch.Tensor,
        model_time_scale: int,
    ):
        if int(model_time_scale) <= 0:
            raise ValueError("model_time_scale must be positive.")
        self.model = model
        self.generator = torch.Generator(device=device).manual_seed(int(source_seed))
        self.normalized_log_rate_edges = make_normalized_log_rate_edges(
            log_rate_edges.to(device)
        )
        self.model_time_scale = float(model_time_scale)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples_per_condition: int,
        num_ode_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, num_quantiles = condition.shape
        num_bins = int(self.normalized_log_rate_edges.numel() - 1)
        repeated_condition = condition[:, None, :, :].expand(
            batch_size,
            num_samples_per_condition,
            condition.shape[1],
            num_quantiles,
        )
        repeated_condition = repeated_condition.reshape(
            batch_size * num_samples_per_condition,
            condition.shape[1],
            num_quantiles,
        )
        source_histograms = sample_standard_dirichlet(
            batch_size * num_samples_per_condition,
            num_bins,
            condition.device,
            torch.float64,
            self.generator,
        )
        state = histogram_to_quantiles(
            source_histograms,
            self.normalized_log_rate_edges,
            num_quantiles=num_quantiles,
        ).to(condition.dtype)

        crossing_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        inspected_pair_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        out_of_domain_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        inspected_value_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        delta_time = 1.0 / float(num_ode_steps)
        lower_bound = self.normalized_log_rate_edges[0].to(dtype=state.dtype)
        upper_bound = self.normalized_log_rate_edges[-1].to(dtype=state.dtype)

        for ode_step_index in range(int(num_ode_steps)):
            flow_time = float(ode_step_index) / float(num_ode_steps)
            flow_times = torch.full(
                (state.shape[0],),
                flow_time,
                device=state.device,
                dtype=state.dtype,
            )
            model_input = torch.cat((state[:, None, :], repeated_condition), dim=1)
            model_times = flow_times * self.model_time_scale
            velocity = unet_sample(
                self.model,
                model_input,
                model_times,
            )[:, 0, :]
            raw_state = state + delta_time * velocity

            crossing_count += (raw_state[:, 1:] < raw_state[:, :-1]).sum().to(torch.float64)
            inspected_pair_count += float(raw_state.shape[0] * (raw_state.shape[1] - 1))
            out_of_domain_count += (
                (raw_state < lower_bound) | (raw_state > upper_bound)
            ).sum().to(torch.float64)
            inspected_value_count += float(raw_state.numel())

            state = _isotonic_projection(
                raw_state.clamp(lower_bound, upper_bound)
            )

        decoded = quantiles_to_histograms(
            state,
            self.normalized_log_rate_edges,
        )
        samples = decoded.reshape(batch_size, num_samples_per_condition, num_bins)
        return samples, {
            "pre_isotonic_projection_crossing_count": crossing_count,
            "pre_isotonic_projection_inspected_pair_count": inspected_pair_count,
            "pre_clipping_out_of_domain_count": out_of_domain_count,
            "pre_clipping_inspected_value_count": inspected_value_count,
        }
