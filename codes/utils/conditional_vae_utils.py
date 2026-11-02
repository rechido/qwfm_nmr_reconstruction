from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from diffusers import UNet1DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from utils.unet_config import (
    NETWORK_ARCHITECTURE_NAME,
    get_network_architecture_signature,
    get_unet_config,
    unet_sample,
    validate_unet_config,
)


CONDITIONAL_VAE_ARCHITECTURE_NAME = "conditional_vae_diffusers_unet1d_decoder_v1"
CONDITIONAL_VAE_INPUT_REPRESENTATION = (
    "posterior_encoder_target_plus_complete_condition_summary_and_"
    "latent_rate_query_plus_broadcast_condition_v1"
)
CONDITIONAL_VAE_OUTPUT_PARAMETERIZATION = "softmax_simplex_v1"
CONDITIONAL_VAE_PRIOR = "isotropic_standard_normal_v1"
CONDITIONAL_VAE_TIMESTEP = 0.0
DEFAULT_ENCODER_HIDDEN_DIM = 256
DEFAULT_ENCODER_NUM_LAYERS = 4


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


class ConditionalVAEModel(ModelMixin, ConfigMixin):
    """Conditional VAE with the benchmark Diffusers UNet1DModel as its decoder.

    The inference network must produce a global latent distribution and is
    therefore an MLP over the target histogram and the non-redundant condition
    vector.  The generative network is exactly the stock U-Net backbone
    used by the FM and deterministic-regression experiments.  Its single state
    channel is a normalized rate-position query plus a learned projection of
    ``z``; all measurement, echo-time, and noise channels are unchanged.
    """

    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int,
        num_measurements: int,
        latent_dim: int,
        encoder_hidden_dim: int = DEFAULT_ENCODER_HIDDEN_DIM,
        encoder_num_layers: int = DEFAULT_ENCODER_NUM_LAYERS,
        architecture_name: str = CONDITIONAL_VAE_ARCHITECTURE_NAME,
        decoder_architecture_name: str = NETWORK_ARCHITECTURE_NAME,
    ):
        super().__init__()
        sample_size = _positive_integer(sample_size, name="sample_size")
        num_measurements = _positive_integer(num_measurements, name="num_measurements")
        latent_dim = _positive_integer(latent_dim, name="latent_dim")
        encoder_hidden_dim = _positive_integer(
            encoder_hidden_dim, name="encoder_hidden_dim"
        )
        encoder_num_layers = _positive_integer(
            encoder_num_layers, name="encoder_num_layers"
        )
        if sample_size < 8:
            raise ValueError("sample_size must be at least eight.")
        if num_measurements < 2:
            raise ValueError("num_measurements must be at least two.")
        if architecture_name != CONDITIONAL_VAE_ARCHITECTURE_NAME:
            raise ValueError(
                "Conditional-VAE architecture mismatch: "
                f"got {architecture_name!r}, required "
                f"{CONDITIONAL_VAE_ARCHITECTURE_NAME!r}."
            )
        if decoder_architecture_name != NETWORK_ARCHITECTURE_NAME:
            raise ValueError(
                "Decoder architecture mismatch: "
                f"got {decoder_architecture_name!r}, required "
                f"{NETWORK_ARCHITECTURE_NAME!r}."
            )

        self.sample_size = sample_size
        self.num_measurements = num_measurements
        self.latent_dim = latent_dim
        condition_channels = 2 * num_measurements + 1
        encoder_input_dim = sample_size + condition_channels

        encoder_layers: list[nn.Module] = []
        input_dim = encoder_input_dim
        for _ in range(encoder_num_layers):
            encoder_layers.extend(
                (
                    nn.Linear(input_dim, encoder_hidden_dim),
                    nn.LayerNorm(encoder_hidden_dim),
                    nn.SiLU(),
                )
            )
            input_dim = encoder_hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.posterior_mean = nn.Linear(encoder_hidden_dim, latent_dim)
        self.posterior_log_variance = nn.Linear(encoder_hidden_dim, latent_dim)

        self.latent_to_rate_query = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(encoder_hidden_dim, sample_size),
        )
        # The decoder itself is identical to the benchmark U-Net.  The extra
        # channel relative to ``condition`` is its state/query channel.
        self.decoder = UNet1DModel(
            **get_unet_config(sample_size=sample_size, num_measurements=num_measurements)
        )
        normalized_positions = (
            torch.arange(sample_size, dtype=torch.float32) + 0.5
        ) / float(sample_size)
        self.register_buffer(
            "normalized_rate_positions",
            normalized_positions,
            persistent=False,
        )

    @property
    def condition_channels(self) -> int:
        return 2 * int(self.config.num_measurements) + 1

    def _validate_condition(self, condition: torch.Tensor) -> None:
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor.")
        if condition.ndim != 3:
            raise ValueError(
                "condition must have shape [batch, channels, bins], "
                f"got {tuple(condition.shape)}."
            )
        expected = (
            condition.shape[0],
            self.condition_channels,
            int(self.config.sample_size),
        )
        if tuple(condition.shape) != expected:
            raise ValueError(
                f"condition has shape {tuple(condition.shape)}, expected {expected}."
            )
        if not condition.is_floating_point():
            raise TypeError("condition must have a floating-point dtype.")

    def encode(
        self,
        target_histograms: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return diagonal-Gaussian posterior mean and log variance."""

        self._validate_condition(condition)
        if not isinstance(target_histograms, torch.Tensor):
            raise TypeError("target_histograms must be a torch.Tensor.")
        expected = (condition.shape[0], int(self.config.sample_size))
        if tuple(target_histograms.shape) != expected:
            raise ValueError(
                "target_histograms must have shape [batch, bins], "
                f"got {tuple(target_histograms.shape)}, expected {expected}."
            )
        if not target_histograms.is_floating_point():
            raise TypeError("target_histograms must have a floating-point dtype.")
        condition_summary = condition[:, :, 0]
        encoder_input = torch.cat(
            (target_histograms.to(dtype=condition.dtype), condition_summary),
            dim=-1,
        )
        hidden = self.encoder(encoder_input)
        mean = self.posterior_mean(hidden)
        # A bounded log variance avoids overflow under mixed precision without
        # changing the diagonal-Gaussian posterior family.
        log_variance = self.posterior_log_variance(hidden).clamp(-20.0, 10.0)
        return mean, log_variance

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if mean.shape != log_variance.shape:
            raise ValueError("mean and log_variance must have identical shapes.")
        noise = torch.randn(
            mean.shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        return mean + torch.exp(0.5 * log_variance) * noise

    def decode(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Decode ``z`` to unit-simplex spectra of shape ``[batch, bins]``."""

        self._validate_condition(condition)
        expected = (condition.shape[0], int(self.config.latent_dim))
        if tuple(latent.shape) != expected:
            raise ValueError(
                f"latent has shape {tuple(latent.shape)}, expected {expected}."
            )
        latent = latent.to(device=condition.device, dtype=condition.dtype)
        latent_query = self.latent_to_rate_query(latent)
        positions = self.normalized_rate_positions.to(
            device=condition.device,
            dtype=condition.dtype,
        )[None, :]
        query_channel = (positions + latent_query)[:, None, :]
        model_input = torch.cat((query_channel, condition), dim=1)
        timesteps = torch.full(
            (condition.shape[0],),
            CONDITIONAL_VAE_TIMESTEP,
            device=condition.device,
            dtype=condition.dtype,
        )
        logits = unet_sample(self.decoder, model_input, timesteps)[:, 0, :]
        softmax_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )
        return torch.softmax(logits.to(dtype=softmax_dtype), dim=-1)

    def forward(
        self,
        target_histograms: torch.Tensor,
        condition: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(target_histograms, condition)
        latent = self.reparameterize(mean, log_variance, generator=generator)
        reconstruction = self.decode(latent, condition)
        return reconstruction, mean, log_variance


def get_conditional_vae_architecture_signature(
    *,
    sample_size: int,
    num_measurements: int,
    latent_dim: int,
    encoder_hidden_dim: int = DEFAULT_ENCODER_HIDDEN_DIM,
    encoder_num_layers: int = DEFAULT_ENCODER_NUM_LAYERS,
) -> dict[str, Any]:
    sample_size = _positive_integer(sample_size, name="sample_size")
    num_measurements = _positive_integer(num_measurements, name="num_measurements")
    latent_dim = _positive_integer(latent_dim, name="latent_dim")
    encoder_hidden_dim = _positive_integer(
        encoder_hidden_dim, name="encoder_hidden_dim"
    )
    encoder_num_layers = _positive_integer(
        encoder_num_layers, name="encoder_num_layers"
    )
    if sample_size < 8:
        raise ValueError("sample_size must be at least eight.")
    return {
        "architecture_name": CONDITIONAL_VAE_ARCHITECTURE_NAME,
        "model_class": ConditionalVAEModel.__name__,
        "sample_size": sample_size,
        "num_measurements": num_measurements,
        "condition_channels": 2 * num_measurements + 1,
        "latent_dim": latent_dim,
        "prior": CONDITIONAL_VAE_PRIOR,
        "encoder": {
            "type": "target_condition_mlp_diagonal_gaussian",
            "hidden_dim": encoder_hidden_dim,
            "num_layers": encoder_num_layers,
            "activation": "silu",
            "normalization": "layer_norm",
        },
        "decoder": {
            **get_network_architecture_signature(
                sample_size=sample_size, num_measurements=num_measurements,
            ),
            "type": UNet1DModel.__name__,
            "latent_query_channels": 1,
        },
        "output_parameterization": CONDITIONAL_VAE_OUTPUT_PARAMETERIZATION,
    }


def create_conditional_vae_model(
    *,
    sample_size: int,
    num_measurements: int,
    latent_dim: int,
    encoder_hidden_dim: int = DEFAULT_ENCODER_HIDDEN_DIM,
    encoder_num_layers: int = DEFAULT_ENCODER_NUM_LAYERS,
) -> ConditionalVAEModel:
    return ConditionalVAEModel(
        sample_size=sample_size,
        num_measurements=num_measurements,
        latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_layers=encoder_num_layers,
    )


def _config_mapping(model: ConditionalVAEModel) -> Mapping[str, Any]:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "get"):
        raise TypeError("Conditional VAE does not expose a Diffusers config mapping.")
    return config


def validate_conditional_vae_model_config(
    model: ConditionalVAEModel,
    *,
    sample_size: int,
    num_measurements: int,
    latent_dim: int,
    encoder_hidden_dim: int = DEFAULT_ENCODER_HIDDEN_DIM,
    encoder_num_layers: int = DEFAULT_ENCODER_NUM_LAYERS,
) -> None:
    if not isinstance(model, ConditionalVAEModel):
        raise TypeError(
            "Incompatible model class: "
            f"got {type(model).__name__}, required {ConditionalVAEModel.__name__}."
        )
    config = _config_mapping(model)
    expected = get_conditional_vae_architecture_signature(
        sample_size=sample_size,
        num_measurements=num_measurements,
        latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_layers=encoder_num_layers,
    )
    expected_config = {
        "architecture_name": expected["architecture_name"],
        "decoder_architecture_name": NETWORK_ARCHITECTURE_NAME,
        "sample_size": sample_size,
        "num_measurements": num_measurements,
        "latent_dim": latent_dim,
        "encoder_hidden_dim": encoder_hidden_dim,
        "encoder_num_layers": encoder_num_layers,
    }
    mismatches = [
        f"{name}: checkpoint={config.get(name)!r}, required={value!r}"
        for name, value in expected_config.items()
        if config.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "Conditional-VAE checkpoint architecture mismatch; " + "; ".join(mismatches)
        )
    validate_unet_config(
        model.decoder, sample_size=sample_size, num_measurements=num_measurements,
    )


def load_conditional_vae_pretrained(path: str | Path) -> ConditionalVAEModel:
    model_path = Path(path)
    loaded_config = ConditionalVAEModel.load_config(model_path)
    if loaded_config.get("_class_name") != ConditionalVAEModel.__name__:
        raise ValueError(
            "Checkpoint model class mismatch: "
            f"checkpoint={loaded_config.get('_class_name')!r}, "
            f"required={ConditionalVAEModel.__name__!r}."
        )
    if loaded_config.get("architecture_name") != CONDITIONAL_VAE_ARCHITECTURE_NAME:
        raise ValueError("Checkpoint conditional-VAE architecture signature mismatch.")
    return ConditionalVAEModel.from_pretrained(model_path)


__all__ = [
    "CONDITIONAL_VAE_ARCHITECTURE_NAME",
    "CONDITIONAL_VAE_INPUT_REPRESENTATION",
    "CONDITIONAL_VAE_OUTPUT_PARAMETERIZATION",
    "CONDITIONAL_VAE_PRIOR",
    "CONDITIONAL_VAE_TIMESTEP",
    "ConditionalVAEModel",
    "create_conditional_vae_model",
    "get_conditional_vae_architecture_signature",
    "load_conditional_vae_pretrained",
    "validate_conditional_vae_model_config",
]
