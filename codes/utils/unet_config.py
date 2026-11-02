"""Shared configuration and checkpoint checks for the stock Diffusers U-Net.

All neural-network layers and the forward pass belong to ``UNet1DModel``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from diffusers import UNet1DModel


NETWORK_ARCHITECTURE_NAME = "diffusers_unet1d_64x4_reslayers3_v1"
UNET_CONFIG = {
    "out_channels": 1,
    "extra_in_channels": 0,
    "time_embedding_type": "positional",
    "use_timestep_embedding": True,
    "flip_sin_to_cos": True,
    "freq_shift": 0.0,
    "down_block_types": ("DownResnetBlock1D",) * 4,
    "up_block_types": ("UpResnetBlock1D",) * 4,
    "mid_block_type": "MidResTemporalBlock1D",
    "out_block_type": "OutConv1DBlock",
    "block_out_channels": (64, 64, 64, 64),
    # Diffusers temporal blocks contain one initial residual layer plus this many.
    "layers_per_block": 2,
    "norm_num_groups": 8,
    "act_fn": "silu",
    "downsample_each_block": False,
}
DOWNSAMPLE_FACTOR = 8


def unet_sample(model: Any, inputs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Run the stock U-Net with right zero padding, then retain physical bins.

    Padding belongs only to the convolutional network: targets, flow states,
    losses, and simplex normalization retain their original length. The
    divisible-by-eight path calls the model without changing its input/output.
    Calling the supplied model (rather than an unwrapped module) also preserves
    Accelerate/DDP forward hooks.
    """

    length = inputs.shape[-1]
    padding = (-length) % DOWNSAMPLE_FACTOR
    if not padding:
        return model(inputs, timesteps).sample
    padded = F.pad(inputs, (0, padding), mode="constant", value=0.0)
    return model(padded, timesteps).sample[..., :length]


def get_unet_config(*, sample_size: int, num_measurements: int) -> dict[str, Any]:
    """Return constructor arguments shared by FM, regression, and the CVAE decoder."""
    for name, value in (("sample_size", sample_size), ("num_measurements", num_measurements)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if sample_size < DOWNSAMPLE_FACTOR:
        raise ValueError(f"sample_size must be at least {DOWNSAMPLE_FACTOR}.")
    if num_measurements < 2:
        raise ValueError("num_measurements must be at least two.")
    return {
        **UNET_CONFIG,
        "sample_size": sample_size + (-sample_size) % DOWNSAMPLE_FACTOR,
        "in_channels": 2 * num_measurements + 2,
    }


def get_network_architecture_signature(
    *, sample_size: int, num_measurements: int,
) -> dict[str, Any]:
    config = get_unet_config(sample_size=sample_size, num_measurements=num_measurements)
    signature = {
        "architecture_name": NETWORK_ARCHITECTURE_NAME,
        "model_class": UNet1DModel.__name__,
        "input_tensor_layout": "batch_channels_length",
        "sample_geometry": [sample_size],
        **{key: list(value) if isinstance(value, tuple) else value for key, value in config.items()},
        "residual_layers_per_block": config["layers_per_block"] + 1,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "num_downsamples": 3,
    }
    if config["sample_size"] != sample_size:
        signature["input_padding"] = {
            "mode": "constant_zero",
            "side": "right",
            "physical_size": sample_size,
            "network_size": config["sample_size"],
            "crop_output_before_loss_or_normalization": True,
        }
    return signature


def validate_unet_config(
    model: UNet1DModel, *, sample_size: int, num_measurements: int,
) -> None:
    """Reject checkpoints whose stock U-Net configuration differs from this experiment."""
    if not isinstance(model, UNet1DModel):
        raise TypeError(f"Expected UNet1DModel, got {type(model).__name__}.")
    expected = get_unet_config(sample_size=sample_size, num_measurements=num_measurements)
    mismatches = []
    for name, value in expected.items():
        actual = model.config.get(name)
        if isinstance(value, tuple) and isinstance(actual, (tuple, list)):
            actual = tuple(actual)
        if actual != value:
            mismatches.append(f"{name}: checkpoint={actual!r}, required={value!r}")
    if mismatches:
        raise ValueError("Checkpoint U-Net configuration mismatch; " + "; ".join(mismatches))
