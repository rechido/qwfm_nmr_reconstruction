from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.nmr_utils import sample_standard_dirichlet
from utils.unet_config import unet_sample


class HistogramLinearTrainer:
    """Concrete Trainer for simplex histogram linear flow matching."""

    def __init__(self, model_time_scale: int):
        if int(model_time_scale) <= 0:
            raise ValueError("model_time_scale must be positive.")
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
        source_histograms = sample_standard_dirichlet(
            batch_size,
            num_bins,
            target_histograms.device,
            target_histograms.dtype,
            # An explicit generator keeps fixed validation probes reproducible.
            generator=generator,
        )
        flow_times = torch.rand(
            batch_size,
            device=target_histograms.device,
            dtype=target_histograms.dtype,
            generator=generator,
        )
        broadcast_time = flow_times[:, None]
        state = (
            (1.0 - broadcast_time) * source_histograms
            + broadcast_time * target_histograms
        )
        target_velocity = target_histograms - source_histograms

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
            # Multimode endpoint views use C * p(mode | y), whose expectation
            # over the C flattened rows is one.  Do not renormalize each random
            # mini-batch: doing so would bias groups with highly variable mode
            # probabilities.
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
