from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.nmr_utils import sample_standard_dirichlet
from utils.unet_config import unet_sample


class HistogramLinearSampler:
    """Explicit Euler sampler in histogram coordinates."""

    def __init__(
        self,
        model,
        device: torch.device,
        source_seed: int,
        model_time_scale: int,
    ):
        if int(model_time_scale) <= 0:
            raise ValueError("model_time_scale must be positive.")
        self.model = model
        self.generator = torch.Generator(device=device).manual_seed(int(source_seed))
        self.model_time_scale = float(model_time_scale)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples_per_condition: int,
        num_ode_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, num_bins = condition.shape
        repeated_condition = condition[:, None, :, :].expand(
            batch_size,
            num_samples_per_condition,
            condition.shape[1],
            num_bins,
        )
        repeated_condition = repeated_condition.reshape(
            batch_size * num_samples_per_condition,
            condition.shape[1],
            num_bins,
        )
        state = sample_standard_dirichlet(
            batch_size * num_samples_per_condition,
            num_bins,
            condition.device,
            condition.dtype,
            self.generator,
        )

        negative_value_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        inspected_value_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        mass_error_sum = torch.zeros((), device=condition.device, dtype=torch.float64)
        inspected_state_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        zero_mass_state_count = torch.zeros((), device=condition.device, dtype=torch.float64)
        delta_time = 1.0 / float(num_ode_steps)

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

            negative_value_count += (raw_state < 0).sum().to(torch.float64)
            inspected_value_count += float(raw_state.numel())
            raw_mass = raw_state.sum(dim=1)
            mass_error_sum += (raw_mass - 1.0).abs().sum().to(torch.float64)
            inspected_state_count += float(raw_state.shape[0])

            state = raw_state.clamp_min(0.0)
            mass = state.sum(dim=1, keepdim=True)
            valid_mass = mass > 1.0e-12
            zero_mass_state_count += (~valid_mass[:, 0]).sum().to(torch.float64)
            normalized = state / mass.clamp_min(1.0e-12)
            uniform = torch.full_like(state, 1.0 / float(num_bins))
            state = torch.where(valid_mass, normalized, uniform)

        samples = state.reshape(batch_size, num_samples_per_condition, num_bins)
        return samples, {
            "pre_projection_negative_value_count": negative_value_count,
            "pre_projection_inspected_value_count": inspected_value_count,
            "pre_projection_mass_error_sum": mass_error_sum,
            "pre_projection_inspected_state_count": inspected_state_count,
            "zero_mass_state_count": zero_mass_state_count,
        }
