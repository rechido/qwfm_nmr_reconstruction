"""Reuse the trained-model contracts and samplers for independent-pair predictive evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import UNet1DModel

from data.grid import build_nmr_grid
from data.spec import ContinuousNMRSpec
from samplers.conditional_normalizing_flow_sampler import ConditionalNormalizingFlowSampler
from samplers.conditional_vae_sampler import ConditionalVAESampler
from samplers.deterministic_regression_sampler import DeterministicRegressionSampler
from samplers.histogram_linear_sampler import HistogramLinearSampler
from samplers.quantile_wasserstein_sampler import QuantileWassersteinSampler
from utils.conditional_normalizing_flow_utils import (
    load_conditional_flow_model_pretrained,
    validate_conditional_flow_model_config,
)
from utils.conditional_vae_utils import (
    load_conditional_vae_pretrained,
    validate_conditional_vae_model_config,
)
from utils.nmr_utils import make_condition
from utils.unet_config import validate_unet_config


_EVALUATORS = {
    "histogram_linear": "evaluate_histogram_linear",
    "quantile_wasserstein": "evaluate_quantile_wasserstein",
    "conditional_vae": "evaluate_conditional_vae",
    "conditional_normalizing_flow": "evaluate_conditional_normalizing_flow",
    "deterministic_regression": "evaluate_deterministic_regression",
    "deterministic_regression_mse_w1": "evaluate_deterministic_regression",
}
_FLOW_METHODS = frozenset(("histogram_linear", "quantile_wasserstein"))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def checkpoint_identity(checkpoint: Path, model_subfolder: str) -> dict[str, Any]:
    """Bind cached predictions to exact model bytes, rather than a mutable 'latest'."""

    model_directory = checkpoint / model_subfolder
    files = [checkpoint / "training_state.json"]
    files.extend(sorted(path for path in model_directory.rglob("*") if path.is_file()))
    if not files[0].is_file() or len(files) < 2:
        raise FileNotFoundError(f"Incomplete checkpoint model: {model_directory}")
    digests: dict[str, str] = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[path.relative_to(checkpoint).as_posix()] = digest.hexdigest()
    encoded = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "model_subfolder": model_subfolder,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files_sha256": digests,
    }


def inspect_training_model(
    training_dir: Path | str,
    checkpoint_path: Path | str = "latest",
    model_subfolder: str = "unet_ema",
) -> dict[str, Any]:
    """Resolve saved training arguments and run the existing evaluator checks.

    This inspection does not instantiate a network or read any SMC references.
    The saved simulator contract determines the independent-pair proper-score test.
    """

    training_dir = Path(training_dir).expanduser().resolve()
    if model_subfolder not in ("unet", "unet_ema"):
        raise ValueError("model_subfolder must be unet or unet_ema.")
    config = _read_json(training_dir / "training_config.json")
    method = config.get("experiment")
    if method not in _EVALUATORS:
        raise ValueError(f"Unsupported trained-model experiment: {method!r}.")
    specification = config.get("specification")
    if not isinstance(specification, dict):
        raise ValueError("Training configuration lacks the simulator specification.")
    spec = ContinuousNMRSpec.from_dict(specification)
    values = dict(specification)
    values.update(config)
    values.update(
        output_dir=str(training_dir),
        checkpoint_path=str(checkpoint_path),
        model_subfolder=model_subfolder,
    )
    if "model_time_scale" not in values and "fm_num_timesteps" in values:
        values["model_time_scale"] = values["fm_num_timesteps"]
    args = argparse.Namespace(**values)
    evaluator = importlib.import_module(f"evaluate.{_EVALUATORS[method]}")
    checkpoint = evaluator._resolve_checkpoint(args).resolve()
    if method in _FLOW_METHODS:
        config, state = evaluator._validate_saved_training_configuration(args, checkpoint)
    elif method.startswith("deterministic_regression"):
        state = evaluator._validate_checkpoint(args, checkpoint)
    else:
        config, state = evaluator._validate_checkpoint(args, checkpoint)
    architecture = None
    if method == "conditional_normalizing_flow":
        architecture = evaluator._architecture_from_saved_config(config)
    return {
        "method": method,
        "spec": spec,
        "training_config": config,
        "checkpoint": checkpoint,
        "checkpoint_state": state,
        "args": args,
        "deterministic": method.startswith("deterministic_regression"),
        "state_size": int(args.num_quantiles if method == "quantile_wasserstein"
                          else args.num_spectrum_bins),
        "checkpoint_identity": checkpoint_identity(checkpoint, model_subfolder),
        "architecture": architecture,
    }


class PredictiveModel:
    """One inference replica and its persistent, explicitly seeded source stream.

    The caller partitions conditions between ranks and supplies a distinct seed
    per rank/job. No implicit rank offset is added here. Inference preparation
    avoids DDP forward collectives, so rank-local batches can be resumed safely.
    """

    def __init__(
        self,
        model_info: dict[str, Any],
        accelerator: Any,
        source_seed: int,
        max_decode_batch_size: int = 1024,
    ) -> None:
        _validate_seed(source_seed)
        _positive_integer(max_decode_batch_size, "max_decode_batch_size")
        self.info = model_info
        self.method = model_info["method"]
        self.accelerator = accelerator
        self.args = model_info["args"]
        self.deterministic = model_info["deterministic"]
        self.grid = build_nmr_grid(model_info["spec"])
        model_directory = model_info["checkpoint"] / self.args.model_subfolder
        if self.method == "conditional_vae":
            model = load_conditional_vae_pretrained(model_directory)
            validate_conditional_vae_model_config(
                model,
                sample_size=self.args.num_spectrum_bins,
                num_measurements=self.args.num_measurements,
                latent_dim=self.args.latent_dim,
                encoder_hidden_dim=self.args.encoder_hidden_dim,
                encoder_num_layers=self.args.encoder_num_layers,
            )
        elif self.method == "conditional_normalizing_flow":
            model = load_conditional_flow_model_pretrained(model_directory)
            validate_conditional_flow_model_config(model, **model_info["architecture"])
        else:
            model = UNet1DModel.from_pretrained(model_directory)
            validate_unet_config(
                model, sample_size=model_info["state_size"],
                num_measurements=self.args.num_measurements,
            )
        model.eval()
        self.model = accelerator.prepare_model(model, evaluation_mode=True)
        if self.method == "conditional_vae":
            self.sampler = ConditionalVAESampler(
                accelerator.unwrap_model(self.model),
                max_decode_batch_size=max_decode_batch_size,
            )
        elif self.method == "conditional_normalizing_flow":
            self.sampler = ConditionalNormalizingFlowSampler(
                accelerator.unwrap_model(self.model), accelerator.device,
                source_seed=source_seed,
            )
        elif self.method == "quantile_wasserstein":
            self.sampler = QuantileWassersteinSampler(
                self.model, accelerator.device, source_seed=source_seed,
                log_rate_edges=self.grid.log_rate_edges,
                model_time_scale=self.args.model_time_scale,
            )
        elif self.method == "histogram_linear":
            self.sampler = HistogramLinearSampler(
                self.model, accelerator.device, source_seed=source_seed,
                model_time_scale=self.args.model_time_scale,
            )
        elif self.deterministic:
            self.sampler = DeterministicRegressionSampler(self.model, self.grid.log_rate_edges)
        else:
            raise ValueError(f"Unsupported trained-model experiment: {self.method!r}.")
        self.generator = torch.Generator(device=accelerator.device)
        self.set_seed(source_seed)

    def set_seed(self, seed: int) -> None:
        """Reset the actual sampler stream, for reproducible resumable blocks."""

        _validate_seed(seed)
        self.generator.manual_seed(seed)
        if hasattr(self.sampler, "generator"):
            self.sampler.generator.manual_seed(seed)

    @torch.no_grad()
    def sample(
        self,
        measurements: torch.Tensor,
        num_samples: int,
        nfe: int = 1,
        source_seed: int | None = None,
        return_quantiles: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return histograms, optionally paired with QWFM terminal quantiles.

        The default is unchanged: [B,K,N] histograms (one for regression).
        ``return_quantiles=True`` is QWFM-only and returns ``(histograms,
        quantiles)`` with the latter shaped [B,K,L].
        """

        _positive_integer(num_samples, "num_samples")
        _positive_integer(nfe, "nfe")
        if return_quantiles and self.method != "quantile_wasserstein":
            raise ValueError("return_quantiles is only supported by QWFM.")
        if self.method not in _FLOW_METHODS and nfe != 1:
            raise ValueError("Only flow-matching samplers support nfe other than 1.")
        if (not isinstance(measurements, torch.Tensor) or measurements.ndim != 2
                or measurements.shape[0] == 0
                or measurements.shape[1] != self.args.num_measurements):
            raise ValueError("measurements must have shape [B, num_measurements] with B > 0.")
        if source_seed is not None:
            self.set_seed(source_seed)
        condition = make_condition(
            measurements.to(device=self.accelerator.device, dtype=torch.float32),
            self.grid.echo_times,
            output_size=self.info["state_size"],
            noise_standard_deviation=self.args.noise_standard_deviation,
        )
        options: dict[str, Any] = {
            "num_samples_per_condition": 1 if self.deterministic else num_samples,
        }
        if self.method in _FLOW_METHODS:
            options["num_ode_steps"] = nfe
        elif self.method == "conditional_vae":
            options["generator"] = self.generator
        if return_quantiles:
            options["return_quantiles"] = True
        with self.accelerator.autocast():
            samples, extras = self.sampler.sample(condition, **options)
        expected = (measurements.shape[0], 1 if self.deterministic else num_samples,
                    self.args.num_spectrum_bins)
        if tuple(samples.shape) != expected or not bool(torch.isfinite(samples).all()):
            raise RuntimeError("Sampler returned an invalid spectrum tensor.")
        samples = samples.detach().to(dtype=torch.float32)
        if return_quantiles:
            quantiles = extras["quantiles"].detach().to(dtype=torch.float32)
            expected_quantiles = (measurements.shape[0], num_samples, self.info["state_size"])
            if tuple(quantiles.shape) != expected_quantiles or not bool(torch.isfinite(quantiles).all()):
                raise RuntimeError("Sampler returned an invalid quantile tensor.")
            return samples, quantiles
        return samples


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("source_seed must be an integer in [0, 2**63).")


__all__ = ["PredictiveModel", "checkpoint_identity", "inspect_training_model"]
