from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.unet_config import unet_sample
from utils.nmr_utils import (
    histogram_to_quantiles,
    make_normalized_log_rate_edges,
    sample_standard_dirichlet,
)


class QuantileWassersteinTrainer:
    """Concrete Trainer for the 1D quantile-Wasserstein geodesic path."""

    def __init__(
        self,
        log_rate_edges: torch.Tensor,
        num_quantiles: int,
        model_time_scale: int,
    ):
        if int(num_quantiles) <= 0:
            raise ValueError("num_quantiles must be positive.")
        if int(model_time_scale) <= 0:
            raise ValueError("model_time_scale must be positive.")
        self.normalized_log_rate_edges = make_normalized_log_rate_edges(
            log_rate_edges
        )
        self.num_quantiles = int(num_quantiles)
        self.model_time_scale = float(model_time_scale)

    def compute_loss(
        self,
        model,
        target_histograms: torch.Tensor,
        condition: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_bins = target_histograms.shape
        if condition.shape[-1] != self.num_quantiles:
            raise ValueError("Condition length must equal num_quantiles.")
        source_histograms = sample_standard_dirichlet(
            batch_size,
            num_bins,
            target_histograms.device,
            torch.float64,
            # An explicit generator keeps fixed validation probes reproducible.
            generator=generator,
        )
        source_quantiles = histogram_to_quantiles(
            source_histograms,
            self.normalized_log_rate_edges,
            num_quantiles=self.num_quantiles,
        ).to(target_histograms.dtype)
        target_quantiles = histogram_to_quantiles(
            target_histograms.to(torch.float64),
            self.normalized_log_rate_edges,
            num_quantiles=self.num_quantiles,
        ).to(target_histograms.dtype)

        flow_times = torch.rand(
            batch_size,
            device=target_histograms.device,
            dtype=target_histograms.dtype,
            generator=generator,
        )
        broadcast_time = flow_times[:, None]
        state = (
            (1.0 - broadcast_time) * source_quantiles
            + broadcast_time * target_quantiles
        )
        target_velocity = target_quantiles - source_quantiles

        model_input = torch.cat((state[:, None, :], condition), dim=1)
        model_times = flow_times * self.model_time_scale
        predicted_velocity = unet_sample(
            model,
            model_input,
            model_times,
        )[:, 0, :]
        squared_error = (predicted_velocity - target_velocity).square()
        if sample_weights is None:
            loss = squared_error.mean()
            coordinate_mse = squared_error.mean(dim=0)
            coordinate_variance = target_velocity.var(dim=0, unbiased=False)
        else:
            if sample_weights.ndim != 1 or sample_weights.shape[0] != batch_size:
                raise ValueError("sample_weights must have shape [batch].")
            if not bool(torch.isfinite(sample_weights).all()) or bool(
                (sample_weights < 0.0).any()
            ):
                raise ValueError("sample_weights must be finite and nonnegative.")
            weights = sample_weights.to(
                device=target_histograms.device,
                dtype=target_histograms.dtype,
            )
            loss = (squared_error.mean(dim=-1) * weights).mean()
            weight_sum = weights.sum().clamp_min(1.0e-12)
            coordinate_mse = (
                squared_error * weights[:, None]
            ).sum(dim=0) / weight_sum
            weighted_mean = (
                target_velocity * weights[:, None]
            ).sum(dim=0) / weight_sum
            coordinate_variance = (
                (target_velocity - weighted_mean).square() * weights[:, None]
            ).sum(dim=0) / weight_sum
        normalized_velocity_mse = (
            coordinate_mse / coordinate_variance.clamp_min(1.0e-12)
        ).mean()
        return loss, {
            "loss": loss.detach(),
            "normalized_velocity_mse": normalized_velocity_mse.detach(),
        }
