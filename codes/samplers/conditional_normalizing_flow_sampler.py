from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.conditional_normalizing_flow_utils import alr_to_simplex


class ConditionalNormalizingFlowSampler:
    """Draw independent conditional spectrum samples by exact flow inversion."""

    def __init__(
        self,
        model,
        device: torch.device,
        *,
        source_seed: int,
    ):
        if isinstance(source_seed, bool) or not isinstance(source_seed, int):
            raise TypeError("source_seed must be an integer.")
        if not 0 <= source_seed < 2**63:
            raise ValueError("source_seed must lie in [0, 2**63).")
        self.model = model
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(source_seed)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples_per_condition: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if condition.ndim != 3:
            raise ValueError(
                "condition must have shape [batch, channels, bins], "
                f"got {tuple(condition.shape)}."
            )
        if (
            isinstance(num_samples_per_condition, bool)
            or not isinstance(num_samples_per_condition, int)
            or num_samples_per_condition <= 0
        ):
            raise ValueError("num_samples_per_condition must be positive.")
        if condition.device != self.device:
            raise ValueError(
                f"condition is on {condition.device}, sampler is on {self.device}."
            )

        batch_size = condition.shape[0]
        repeated_condition = condition.repeat_interleave(
            num_samples_per_condition, dim=0
        )
        # Evaluation passes an unwrapped model, but accepting a DDP-like wrapper
        # here prevents the same custom-method proxy failure fixed in training.
        flow_model = getattr(self.model, "module", self.model)
        model_dtype = next(flow_model.parameters()).dtype
        base = torch.randn(
            batch_size * num_samples_per_condition,
            int(flow_model.alr_dimension),
            device=self.device,
            dtype=model_dtype,
            generator=self.generator,
        )
        alr_coordinates = flow_model.decode(
            base, repeated_condition.to(dtype=model_dtype)
        )
        flat_samples = alr_to_simplex(alr_coordinates).to(torch.float32)
        samples = flat_samples.reshape(
            batch_size,
            num_samples_per_condition,
            condition.shape[-1],
        )

        mass_error = (samples.sum(dim=-1) - 1.0).abs()
        diagnostics = {
            "base_draw_count": torch.tensor(
                float(base.shape[0]), device=self.device, dtype=torch.float64
            ),
            "inverse_coupling_transform_count": torch.tensor(
                float(base.shape[0] * len(flow_model.coupling_layers)),
                device=self.device,
                dtype=torch.float64,
            ),
            "generated_state_count": torch.tensor(
                float(base.shape[0]), device=self.device, dtype=torch.float64
            ),
            "generated_value_count": torch.tensor(
                float(samples.numel()), device=self.device, dtype=torch.float64
            ),
            "negative_value_count": (samples < 0.0).sum().to(torch.float64),
            "nonfinite_value_count": (~torch.isfinite(samples)).sum().to(torch.float64),
            "mass_absolute_error_sum": mass_error.sum().to(torch.float64),
            "alr_absolute_value_sum": alr_coordinates.abs().sum().to(torch.float64),
            "alr_value_count": torch.tensor(
                float(alr_coordinates.numel()),
                device=self.device,
                dtype=torch.float64,
            ),
        }
        return samples, diagnostics


__all__ = ["ConditionalNormalizingFlowSampler"]
