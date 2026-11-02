"""Names and saved-configuration contracts for supported learned models."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


VARIANTS = {
    "histogram_linear": ("histogram_linear", "histogram_linear_{shape}_unet64x4_t1024", None, None),
    "quantile_wasserstein_l128": ("quantile_wasserstein", "quantile_wasserstein_{shape}_l128_unet64x4_t1024", "num_quantiles", 128),
    "quantile_wasserstein_l256": ("quantile_wasserstein", "quantile_wasserstein_{shape}_l256_unet64x4_t1024", "num_quantiles", 256),
    "quantile_wasserstein_l512": ("quantile_wasserstein", "quantile_wasserstein_{shape}_l512_unet64x4_t1024", "num_quantiles", 512),
    "deterministic_regression": ("deterministic_regression", "deterministic_regression_{shape}_unet64x4", None, None),
    "deterministic_regression_mse_w1": ("deterministic_regression_mse_w1", "deterministic_regression_mse_w1_{shape}_unet64x4", None, None),
    "conditional_normalizing_flow": ("conditional_normalizing_flow", "conditional_normalizing_flow_{shape}_realnvp8x256", None, None),
    "conditional_vae_z16": ("conditional_vae", "conditional_vae_{shape}_unet64x4_z16", "latent_dim", 16),
    "conditional_vae_z32": ("conditional_vae", "conditional_vae_{shape}_unet64x4_z32", "latent_dim", 32),
}


FLOW_METHODS = {"histogram_linear", "quantile_wasserstein"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_variant(config: dict[str, Any], variant: str) -> None:
    method, _, field, expected = VARIANTS[variant]
    if config.get("experiment") != method:
        raise ValueError(f"{variant} requires experiment={method} in the saved training configuration.")
    if field is not None and config.get(field) != expected:
        raise ValueError(f"{variant} requires {field}={expected} in the saved training configuration.")


def resolve_training_dir(
    experiment_dir: Path, variant: str, training_dir: Path | None = None,
    *, require_config: bool = True,
) -> Path:
    experiment_dir = experiment_dir.expanduser().resolve()
    if training_dir is None:
        match = re.fullmatch(r"m(32|64)n128", experiment_dir.name)
        if match is None:
            raise ValueError("Automatic model names support m32n128/m64n128; supply --training_dir for a custom experiment.")
        shape = f"m{match.group(1)}_n128"
        result = experiment_dir / VARIANTS[variant][1].format(shape=shape)
    else:
        result = training_dir.expanduser().resolve()
        if result.parent != experiment_dir:
            raise ValueError("--training_dir must be a direct child of --experiment_dir.")
    config_path = result / "training_config.json"
    if require_config or training_dir is not None or config_path.exists():
        validate_variant(_read_json(config_path), variant)
    return result
