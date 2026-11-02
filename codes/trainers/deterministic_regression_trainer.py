from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from utils.deterministic_regression_utils import (
    REGRESSION_LOSS_KINDS,
    deterministic_regression_forward,
    make_normalized_log_rate_positions,
)
from utils.nmr_utils import histogram_w1


class DeterministicRegressionTrainer:
    """MSE trainer for the matched-backbone deterministic histogram baseline.

    Optional ``sample_weights`` are importance weights with the caller-facing
    convention ``C * w`` for a C-branch family whose posterior probabilities
    sum to one.  They intentionally are *not* renormalized by their minibatch
    sum: ``mean(sample_weights * per_sample_mse)`` is the unbiased, mean-one
    weighted objective when branches are sampled uniformly.  Renormalizing by
    a stochastic minibatch sum would instead introduce a ratio-estimator bias.
    """

    def __init__(
        self,
        log_rate_edges: torch.Tensor,
        *,
        loss_kind: str = "mse",
        mse_loss_weight: float = 1.0,
        wasserstein_loss_weight: float = 0.0,
    ):
        if loss_kind not in REGRESSION_LOSS_KINDS:
            raise ValueError(
                f"loss_kind must be one of {REGRESSION_LOSS_KINDS}, got {loss_kind!r}."
            )
        for name, value in (
            ("mse_loss_weight", mse_loss_weight),
            ("wasserstein_loss_weight", wasserstein_loss_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if float(mse_loss_weight) <= 0.0:
            raise ValueError("mse_loss_weight must be positive.")
        if loss_kind == "mse" and float(wasserstein_loss_weight) != 0.0:
            raise ValueError(
                "wasserstein_loss_weight must be zero for the MSE baseline."
            )
        if loss_kind == "mse_w1" and float(wasserstein_loss_weight) <= 0.0:
            raise ValueError("wasserstein_loss_weight must be positive for MSE+W1.")
        self.loss_kind = str(loss_kind)
        self.mse_loss_weight = float(mse_loss_weight)
        self.wasserstein_loss_weight = float(wasserstein_loss_weight)
        self.log_rate_edges = log_rate_edges.detach().to(torch.float64).contiguous()
        self.normalized_log_rate_positions = make_normalized_log_rate_positions(
            log_rate_edges
        )

    def compute_loss(
        self,
        model,
        target_histograms: torch.Tensor,
        condition: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
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
            raise ValueError(
                "target_histograms and condition must have the same batch size."
            )
        if target_histograms.shape[1] != condition.shape[2]:
            raise ValueError(
                "target_histograms and condition must have the same bin count."
            )
        if target_histograms.shape[1] != self.normalized_log_rate_positions.numel():
            raise ValueError(
                "The target bin count does not match the configured log-rate grid."
            )
        if not target_histograms.is_floating_point():
            raise TypeError("target_histograms must have a floating-point dtype.")

        predictions = deterministic_regression_forward(
            model,
            condition,
            self.normalized_log_rate_positions,
        )
        targets = target_histograms.to(dtype=predictions.dtype)
        per_sample_mse = (predictions - targets).square().mean(dim=-1)
        if self.loss_kind == "mse":
            # Preserve the original MSE autograd graph exactly.  W1 remains a
            # useful diagnostic, but it must not be attached through a
            # mathematically redundant zero-weight branch.
            with torch.no_grad():
                per_sample_log_w1 = histogram_w1(
                    predictions.detach(),
                    targets.detach(),
                    self.log_rate_edges,
                )
        else:
            per_sample_log_w1 = histogram_w1(
                predictions,
                targets,
                self.log_rate_edges,
            )

        if sample_weights is None:
            weights = torch.ones_like(per_sample_mse)
        else:
            if not isinstance(sample_weights, torch.Tensor):
                raise TypeError("sample_weights must be a torch.Tensor or None.")
            if (
                sample_weights.ndim != 1
                or sample_weights.shape[0] != predictions.shape[0]
            ):
                raise ValueError(
                    "sample_weights must have shape [batch], "
                    f"got {tuple(sample_weights.shape)} for batch "
                    f"{predictions.shape[0]}."
                )
            weights = sample_weights.detach().to(
                device=predictions.device,
                dtype=predictions.dtype,
            )
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
                raise ValueError("sample_weights must be finite and nonnegative.")
        if self.loss_kind == "mse":
            per_sample_objective = self.mse_loss_weight * per_sample_mse
        else:
            per_sample_objective = (
                self.mse_loss_weight * per_sample_mse
                + self.wasserstein_loss_weight * per_sample_log_w1
            )
        loss = (weights * per_sample_objective).mean()
        squared_l2_error = (predictions - targets).square().sum(dim=-1)
        target_squared_l2 = targets.square().sum(dim=-1).clamp_min(1.0e-12)
        relative_squared_error = squared_l2_error / target_squared_l2

        return loss, {
            "loss": loss.detach(),
            "unweighted_histogram_mse": per_sample_mse.mean().detach(),
            "histogram_log_w1": (weights * per_sample_log_w1).mean().detach(),
            "weighted_mse_loss_component": (
                weights * self.mse_loss_weight * per_sample_mse
            )
            .mean()
            .detach(),
            "weighted_wasserstein_loss_component": (
                weights * self.wasserstein_loss_weight * per_sample_log_w1
            )
            .mean()
            .detach(),
            "normalized_histogram_mse": (weights * relative_squared_error)
            .mean()
            .detach(),
            "prediction_nrmse_mean": (weights * relative_squared_error.sqrt())
            .mean()
            .detach(),
            "sample_weight_mean": weights.mean().detach(),
        }


__all__ = ["REGRESSION_LOSS_KINDS", "DeterministicRegressionTrainer"]
