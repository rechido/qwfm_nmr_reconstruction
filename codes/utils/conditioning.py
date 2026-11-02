from __future__ import annotations

import math

import torch


def make_condition(
    measurements: torch.Tensor,
    echo_times: torch.Tensor,
    *,
    output_size: int,
    noise_standard_deviation: float,
) -> torch.Tensor:
    """Broadcast the complete measurement, acquisition, and noise condition."""

    if not isinstance(measurements, torch.Tensor) or measurements.ndim != 2:
        raise ValueError("measurements must have shape [batch, measurements].")
    if not measurements.is_floating_point() or not bool(torch.isfinite(measurements).all()):
        raise ValueError("measurements must be finite and floating point.")
    if not isinstance(echo_times, torch.Tensor) or echo_times.ndim != 1:
        raise ValueError("echo_times must be one-dimensional.")
    if int(echo_times.shape[0]) != int(measurements.shape[1]):
        raise ValueError("echo_times must match the measurement dimension.")
    if not echo_times.is_floating_point() or not bool(torch.isfinite(echo_times).all()):
        raise ValueError("echo_times must be finite and floating point.")
    if bool((echo_times <= 0.0).any()) or bool((torch.diff(echo_times) <= 0.0).any()):
        raise ValueError("echo_times must be positive and strictly increasing.")
    if isinstance(output_size, bool) or not isinstance(output_size, int) or output_size < 2:
        raise ValueError("output_size must be an integer of at least two.")
    if (
        isinstance(noise_standard_deviation, bool)
        or not isinstance(noise_standard_deviation, (int, float))
        or not math.isfinite(float(noise_standard_deviation))
        or float(noise_standard_deviation) <= 0.0
    ):
        raise ValueError("noise_standard_deviation must be finite and positive.")
    measurements = measurements.contiguous()
    measurement_channels = measurements[:, :, None].expand(-1, -1, output_size)
    log_echo = echo_times.to(device=measurements.device, dtype=measurements.dtype).log()
    normalized_echo = (log_echo - log_echo[0]) / (log_echo[-1] - log_echo[0])
    echo_channels = normalized_echo[None, :, None].expand(
        measurements.shape[0], -1, output_size
    )
    sigma_channel = torch.full(
        (measurements.shape[0], 1, output_size),
        float(noise_standard_deviation),
        device=measurements.device,
        dtype=measurements.dtype,
    )
    return torch.cat((measurement_channels, echo_channels, sigma_channel), dim=1)


__all__ = ["make_condition"]

