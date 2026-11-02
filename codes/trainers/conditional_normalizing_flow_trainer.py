from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from utils.conditional_normalizing_flow_utils import simplex_to_alr


class ConditionalNormalizingFlowTrainer:
    """Posterior-weighted maximum-likelihood trainer for the conditional flow.

    The grouped dataset exposes every posterior branch uniformly and supplies
    the unbiased importance weight ``C * p(branch | y)``.  The same weighting
    convention used by deterministic regression is therefore applied to NLL.

    Sparse target spectra represent point masses in ALR coordinates.  A fixed
    Gaussian perturbation in ALR space is used as explicit dequantization so
    conditional maximum likelihood has a finite-width target distribution.
    """

    def __init__(
        self,
        *,
        interior_mass: float,
        alr_dequantization_std: float,
    ):
        if not 0.0 < float(interior_mass) < 1.0:
            raise ValueError("interior_mass must lie in (0, 1).")
        if (
            not math.isfinite(float(alr_dequantization_std))
            or float(alr_dequantization_std) < 0.0
        ):
            raise ValueError("alr_dequantization_std must be finite and nonnegative.")
        self.interior_mass = float(interior_mass)
        self.alr_dequantization_std = float(alr_dequantization_std)

    def compute_loss(
        self,
        model,
        target_histograms: torch.Tensor,
        condition: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if target_histograms.ndim != 2:
            raise ValueError(
                "target_histograms must have shape [batch, bins], "
                f"got {tuple(target_histograms.shape)}."
            )
        if condition.ndim != 3:
            raise ValueError(
                "condition must have shape [batch, channels, bins], "
                f"got {tuple(condition.shape)}."
            )
        if target_histograms.shape[0] != condition.shape[0]:
            raise ValueError("Target and condition batch sizes must match.")
        if target_histograms.shape[1] != condition.shape[2]:
            raise ValueError("Target and condition bin counts must match.")

        # ALR/log-density arithmetic remains float32 when the outer training
        # loop uses fp16/bf16 autocast.  Coupling MLP operations can still be
        # autocast by Accelerator.
        coordinates = simplex_to_alr(
            target_histograms,
            interior_mass=self.interior_mass,
        )
        if self.alr_dequantization_std:
            # Preserve the original objective while permitting deterministic
            # fixed validation through an explicit random generator.
            coordinates = coordinates + self.alr_dequantization_std * torch.randn(
                coordinates.shape,
                device=coordinates.device,
                dtype=coordinates.dtype,
                generator=generator,
            )

        # Call the module rather than its custom ``log_prob`` method.  In
        # distributed training ``model`` is a DistributedDataParallel wrapper,
        # whose forward path owns gradient synchronization/no-sync handling and
        # does not expose arbitrary methods of the wrapped module.
        log_probability, latent, logdet = model(
            coordinates,
            condition,
            return_diagnostics=True,
        )
        if not bool(torch.isfinite(log_probability).all()):
            raise FloatingPointError("CNF produced a non-finite log probability.")
        per_sample_nll = -log_probability
        if sample_weights is None:
            weights = torch.ones_like(per_sample_nll)
        else:
            if not isinstance(sample_weights, torch.Tensor):
                raise TypeError("sample_weights must be a torch.Tensor or None.")
            if sample_weights.ndim != 1 or sample_weights.shape != per_sample_nll.shape:
                raise ValueError(
                    "sample_weights must have shape [batch], "
                    f"got {tuple(sample_weights.shape)}."
                )
            weights = sample_weights.detach().to(
                device=per_sample_nll.device,
                dtype=per_sample_nll.dtype,
            )
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
                raise ValueError("sample_weights must be finite and nonnegative.")

        alr_dimension = float(coordinates.shape[-1])
        # Normalize by dimension to keep the optimization scale comparable
        # when the spectrum grid is changed; this does not change the optimum.
        loss = (weights * per_sample_nll).mean() / alr_dimension
        weighted_nll = (weights * per_sample_nll).mean()
        latent_mean = latent.mean(dim=0)
        latent_variance = latent.var(dim=0, unbiased=False)
        return loss, {
            "loss": loss.detach(),
            "conditional_nll": weighted_nll.detach(),
            "conditional_nll_per_alr_dimension": (
                weighted_nll / alr_dimension
            ).detach(),
            "conditional_bits_per_alr_dimension": (
                weighted_nll / (alr_dimension * math.log(2.0))
            ).detach(),
            "unweighted_conditional_nll": per_sample_nll.mean().detach(),
            "mean_log_absolute_jacobian_determinant": logdet.mean().detach(),
            "latent_mean_absolute_mean": latent_mean.abs().mean().detach(),
            "latent_variance_absolute_error_mean": (latent_variance - 1.0)
            .abs()
            .mean()
            .detach(),
            "sample_weight_mean": weights.mean().detach(),
        }


__all__ = ["ConditionalNormalizingFlowTrainer"]
