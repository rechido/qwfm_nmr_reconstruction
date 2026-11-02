from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


class ConditionalVAETrainer:
    """Posterior-weighted ELBO-style trainer for simplex histograms.

    A flattened family-mode row carries the same caller-facing objective weight
    ``C * p(mode | y)`` used by deterministic regression and the flow models.
    We deliberately average these weights over the minibatch instead of
    normalizing by their random minibatch sum, avoiding ratio-estimator bias.
    """

    def __init__(self, *, kl_weight: float):
        if (
            isinstance(kl_weight, bool)
            or not isinstance(kl_weight, (int, float))
            or not math.isfinite(float(kl_weight))
            or float(kl_weight) < 0.0
        ):
            raise ValueError("kl_weight must be finite and nonnegative.")
        self.kl_weight = float(kl_weight)

    def compute_loss(
        self,
        model,
        target_histograms: torch.Tensor,
        condition: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        *,
        kl_multiplier: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if target_histograms.ndim != 2:
            raise ValueError(
                "target_histograms must have shape [batch, bins], "
                f"got {tuple(target_histograms.shape)}."
            )
        if condition.ndim != 3:
            raise ValueError(
                "condition must have shape [batch, condition_channels, bins], "
                f"got {tuple(condition.shape)}."
            )
        if target_histograms.shape[0] != condition.shape[0]:
            raise ValueError("target_histograms and condition batch sizes differ.")
        if target_histograms.shape[1] != condition.shape[2]:
            raise ValueError("target_histograms and condition bin counts differ.")
        if (
            isinstance(kl_multiplier, bool)
            or not isinstance(kl_multiplier, (int, float))
            or not math.isfinite(float(kl_multiplier))
            or not 0.0 <= float(kl_multiplier) <= 1.0
        ):
            raise ValueError("kl_multiplier must lie in [0, 1].")

        reconstructions, posterior_mean, posterior_log_variance = model(
            target_histograms,
            condition,
            generator=generator,
        )
        targets = target_histograms.to(dtype=reconstructions.dtype)
        per_sample_mse = (reconstructions - targets).square().mean(dim=-1)
        # Mean over latent coordinates keeps the user-facing KL coefficient
        # comparable when latent_dim is changed in an ablation.
        per_sample_kl = 0.5 * (
            posterior_mean.square()
            + posterior_log_variance.exp()
            - 1.0
            - posterior_log_variance
        ).mean(dim=-1)

        if sample_weights is None:
            weights = torch.ones_like(per_sample_mse)
        else:
            if not isinstance(sample_weights, torch.Tensor):
                raise TypeError("sample_weights must be a torch.Tensor or None.")
            if tuple(sample_weights.shape) != (reconstructions.shape[0],):
                raise ValueError(
                    "sample_weights must have shape [batch], "
                    f"got {tuple(sample_weights.shape)}."
                )
            weights = sample_weights.detach().to(
                device=reconstructions.device,
                dtype=reconstructions.dtype,
            )
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("sample_weights must be finite and nonnegative.")

        effective_kl_weight = self.kl_weight * float(kl_multiplier)
        per_sample_objective = per_sample_mse + effective_kl_weight * per_sample_kl
        loss = (weights * per_sample_objective).mean()

        squared_l2 = (reconstructions - targets).square().sum(dim=-1)
        target_squared_l2 = targets.square().sum(dim=-1).clamp_min(1.0e-12)
        relative_squared_error = squared_l2 / target_squared_l2
        posterior_std = torch.exp(0.5 * posterior_log_variance)
        return loss, {
            "loss": loss.detach(),
            "weighted_reconstruction_mse": (weights * per_sample_mse).mean().detach(),
            "unweighted_reconstruction_mse": per_sample_mse.mean().detach(),
            "weighted_kl_per_latent": (weights * per_sample_kl).mean().detach(),
            "unweighted_kl_per_latent": per_sample_kl.mean().detach(),
            "prediction_nrmse_mean": (weights * relative_squared_error.sqrt())
            .mean()
            .detach(),
            "posterior_std_mean": posterior_std.mean().detach(),
            "sample_weight_mean": weights.mean().detach(),
            "effective_kl_weight": torch.tensor(
                effective_kl_weight,
                device=loss.device,
                dtype=loss.dtype,
            ),
        }


__all__ = ["ConditionalVAETrainer"]
