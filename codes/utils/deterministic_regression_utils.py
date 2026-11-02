from __future__ import annotations

import torch

from .unet_config import unet_sample

DETERMINISTIC_REGRESSION_TIMESTEP = 0.0
DETERMINISTIC_REGRESSION_INPUT_REPRESENTATION = (
    "normalized_log_rate_position_plus_broadcast_condition_v1"
)
DETERMINISTIC_REGRESSION_OUTPUT_PARAMETERIZATION = "softmax_simplex_v1"
REGRESSION_LOSS_KINDS = ("mse", "mse_w1")


def make_normalized_log_rate_positions(
    log_rate_edges: torch.Tensor,
) -> torch.Tensor:
    """Return normalized log-rate bin centers for the regression query channel."""

    if not isinstance(log_rate_edges, torch.Tensor):
        raise TypeError("log_rate_edges must be a torch.Tensor.")
    if log_rate_edges.ndim != 1 or log_rate_edges.numel() < 3:
        raise ValueError(
            "log_rate_edges must be one-dimensional and contain at least three edges."
        )
    if not log_rate_edges.is_floating_point():
        raise TypeError("log_rate_edges must have a floating-point dtype.")

    edges = log_rate_edges.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(edges).all()) or bool((torch.diff(edges) <= 0).any()):
        raise ValueError("log_rate_edges must be finite and strictly increasing.")

    centers = 0.5 * (edges[:-1] + edges[1:])
    positions = (centers - edges[0]) / (edges[-1] - edges[0])
    if not bool(torch.isfinite(positions).all()) or bool(
        ((positions <= 0.0) | (positions >= 1.0)).any()
    ):
        raise RuntimeError("Normalized log-rate positions must lie strictly in (0, 1).")
    return positions.contiguous()


def deterministic_regression_forward(
    model,
    condition: torch.Tensor,
    normalized_log_rate_positions: torch.Tensor,
) -> torch.Tensor:
    """Predict simplex histograms with the shared stock UNet1DModel backbone.

    ``condition`` has the same broadcast representation used by the flow models:
    measurement channels, normalized log-echo channels, and one noise channel.
    The flow state slot is replaced by a fixed normalized log-rate position
    channel.  The otherwise unchanged time-conditioned network is always called
    at timestep zero.  Softmax is evaluated in at least float32 so fp16/bf16
    autocast cannot compromise the simplex normalization.
    """

    if not isinstance(condition, torch.Tensor):
        raise TypeError("condition must be a torch.Tensor.")
    if condition.ndim != 3:
        raise ValueError(
            "condition must have shape [batch, condition_channels, bins], "
            f"got {tuple(condition.shape)}."
        )
    if not condition.is_floating_point():
        raise TypeError("condition must have a floating-point dtype.")
    if not isinstance(normalized_log_rate_positions, torch.Tensor):
        raise TypeError("normalized_log_rate_positions must be a torch.Tensor.")
    if (
        normalized_log_rate_positions.ndim != 1
        or normalized_log_rate_positions.shape[0] != condition.shape[-1]
    ):
        raise ValueError(
            "normalized_log_rate_positions must have shape [bins] matching "
            f"the condition length, got {tuple(normalized_log_rate_positions.shape)} "
            f"for length {condition.shape[-1]}."
        )

    position_channel = normalized_log_rate_positions.to(
        device=condition.device,
        dtype=condition.dtype,
    )[None, None, :].expand(condition.shape[0], 1, condition.shape[-1])
    model_input = torch.cat((position_channel, condition), dim=1)
    fixed_timesteps = torch.full(
        (condition.shape[0],),
        DETERMINISTIC_REGRESSION_TIMESTEP,
        device=condition.device,
        dtype=condition.dtype,
    )
    logits = unet_sample(model, model_input, fixed_timesteps)[:, 0, :]
    softmax_dtype = (
        torch.float32
        if logits.dtype in (torch.float16, torch.bfloat16)
        else logits.dtype
    )
    return torch.softmax(logits.to(dtype=softmax_dtype), dim=-1)
