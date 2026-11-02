from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

try:  # Direct old-style entrypoint execution puts ``codes`` on sys.path.
    from data import ContinuousNMRSpec, build_nmr_grid
    from evaluate.artifact_contract import NORMALIZED_HISTOGRAM_REPRESENTATION
    from evaluate.evaluate_saved_samples import (
        EvaluationSettings,
        evaluate_saved_samples_artifacts,
        validate_evaluation_artifacts,
    )
    from utils.io_utils import atomic_torch_save, atomic_write_json, load_torch_artifact
except ImportError:  # Package imports used by tests.
    from codes.data import ContinuousNMRSpec, build_nmr_grid
    from codes.evaluate.artifact_contract import NORMALIZED_HISTOGRAM_REPRESENTATION
    from codes.evaluate.evaluate_saved_samples import (
        EvaluationSettings,
        evaluate_saved_samples_artifacts,
        validate_evaluation_artifacts,
    )
    from codes.utils.io_utils import (
        atomic_torch_save,
        atomic_write_json,
        load_torch_artifact,
    )


SAMPLING_ARTIFACT_SCHEMA_VERSION = 2


class PosteriorConditionDataset(Dataset):
    """Read-only condition view of one validated posterior-reference bank."""

    def __init__(self, reference: Mapping[str, Any]) -> None:
        self.reference = reference
        self.condition_ids = reference["condition_ids"]
        self.measurements = reference["measurements"]
        self.truth_histograms = reference["truth_histograms"]
        self.grid = reference["grid"]
        if self.condition_ids.ndim != 1 or self.measurements.ndim != 2:
            raise ValueError(
                "Reference condition IDs and measurements have invalid shapes."
            )
        if self.truth_histograms.ndim != 2:
            raise ValueError("Reference truth_histograms must have shape [G,N].")
        if self.condition_ids.shape[0] != self.measurements.shape[0] or (
            self.truth_histograms.shape[0] != self.condition_ids.shape[0]
        ):
            raise ValueError("Reference condition fields have inconsistent lengths.")

    def __len__(self) -> int:
        return int(self.condition_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "condition_id": self.condition_ids[index],
            "measurement": self.measurements[index],
            "target_histogram": self.truth_histograms[index],
        }

    @property
    def echo_times(self) -> torch.Tensor:
        return self.grid["echo_times"]

    @property
    def rate_centers(self) -> torch.Tensor:
        return self.grid["rate_centers"]

    @property
    def log_rate_edges(self) -> torch.Tensor:
        return self.grid["log_rate_edges"]

    @property
    def kernel(self) -> torch.Tensor:
        return self.grid["kernel"]


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("Expected true/false, 1/0, or yes/no.")


def _nested_equal(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return (
            torch.is_tensor(first)
            and torch.is_tensor(second)
            and first.dtype == second.dtype
            and tuple(first.shape) == tuple(second.shape)
            and bool(torch.equal(first.cpu(), second.cpu()))
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_nested_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_nested_equal(a, b) for a, b in zip(first, second))
        )
    return type(first) is type(second) and first == second


def _validate_reference_grid(
    actual: Any, expected: Mapping[str, Any], *, name: str
) -> None:
    """Allow CPU roundoff in rebuilt grids, not in stored-data/cache identity."""

    prefix = f"{name} reference grid differs from the CLI specification"
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise ValueError(f"{prefix}: grid fields do not match.")
    for key, expected_value in expected.items():
        value = actual[key]
        field = f"{prefix}: grid[{key!r}]"
        if not torch.is_tensor(expected_value):
            if not _nested_equal(value, expected_value):
                raise ValueError(f"{field} does not match.")
            continue
        if not torch.is_tensor(value):
            raise ValueError(f"{field} must be a tensor.")
        if value.dtype != expected_value.dtype:
            raise ValueError(
                f"{field} dtype {value.dtype} != expected {expected_value.dtype}."
            )
        if value.shape != expected_value.shape:
            raise ValueError(
                f"{field} shape {tuple(value.shape)} != "
                f"expected {tuple(expected_value.shape)}."
            )
        value, expected_value = value.cpu(), expected_value.cpu()
        if expected_value.is_floating_point():
            if not bool(
                torch.isfinite(value).all() & torch.isfinite(expected_value).all()
            ):
                raise ValueError(f"{field} contains non-finite values.")
            # Match offline evaluation/audit; float64 exp/linspace can vary by CPU.
            if not torch.allclose(value, expected_value, atol=1.0e-12, rtol=0.0):
                difference = float((value - expected_value).abs().max())
                raise ValueError(
                    f"{field} max_abs_difference={difference:.6g} "
                    "exceeds atol=1e-12 (rtol=0)."
                )
        elif not torch.equal(value, expected_value):
            raise ValueError(f"{field} values do not match exactly.")


def load_and_validate_references(
    primary_path: str | Path,
    oracle_path: str | Path,
    spec: ContinuousNMRSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = load_torch_artifact(primary_path)
    oracle = load_torch_artifact(oracle_path)
    if not isinstance(primary, dict) or not isinstance(oracle, dict):
        raise ValueError("Posterior references must be dictionary artifacts.")
    for reference, name in ((primary, "primary"), (oracle, "oracle")):
        if reference.get("protocol_id") != spec.protocol_id:
            raise ValueError(f"{name} reference protocol differs from the CLI spec.")
        if reference.get("specification_hash") != spec.specification_hash:
            raise ValueError(
                f"{name} reference specification hash differs from training/CLI."
            )
        embedded = ContinuousNMRSpec.from_dict(reference.get("specification", {}))
        if embedded.specification_hash != spec.specification_hash:
            raise ValueError(f"{name} reference embeds an inconsistent specification.")
    if int(primary.get("reference_run_id", -1)) == int(
        oracle.get("reference_run_id", -1)
    ):
        raise ValueError("Primary and oracle references must be independent runs.")
    for field in (
        "condition_ids",
        "measurements",
        "truth_histograms",
        "depths_m",
        "observation_metadata",
    ):
        if not _nested_equal(primary.get(field), oracle.get(field)):
            raise ValueError(f"Primary and oracle references differ in {field}.")
    expected_grid = build_nmr_grid(spec).as_artifact()
    _validate_reference_grid(primary.get("grid"), expected_grid, name="Primary")
    _validate_reference_grid(oracle.get("grid"), expected_grid, name="Oracle")
    return primary, oracle


def build_sampling_artifact(
    *,
    spec: ContinuousNMRSpec,
    primary_reference: Mapping[str, Any],
    samples: torch.Tensor,
    method: str,
    training_config: Mapping[str, Any],
    output_dir: str | Path,
    checkpoint: str | Path,
    model_subfolder: str,
    primary_reference_path: str | Path,
    state_size: int,
    deterministic: bool = False,
    sampling_config: Mapping[str, Any],
    source_seed: int,
    sampling_wall_seconds: float,
    inference_settings: Mapping[str, Any],
    sampler_diagnostic_totals: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    samples = samples.detach().cpu().to(torch.float32)
    groups = int(primary_reference["condition_ids"].shape[0])
    if tuple(samples.shape[:1] + samples.shape[2:]) != (groups, spec.num_spectrum_bins):
        raise ValueError("samples must have shape [G,K,N] matching the reference.")
    if not bool(torch.isfinite(samples).all()) or bool((samples < 0).any()):
        raise ValueError("Generated histograms must be finite and nonnegative.")
    mass_error = (samples.sum(dim=-1) - 1.0).abs()
    if float(mass_error.max()) > 2.0e-4:
        raise ValueError("Generated histograms violate the unit-mass contract.")
    unique_counts = torch.tensor(
        [torch.unique(row, dim=0).shape[0] for row in samples], dtype=torch.int64
    )
    artifact = {
        "artifact_schema_version": SAMPLING_ARTIFACT_SCHEMA_VERSION,
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "condition_ids": primary_reference["condition_ids"].cpu().clone(),
        "measurements": primary_reference["measurements"].cpu().clone(),
        "samples": samples,
        "grid": primary_reference["grid"],
        "representation_metadata": {
            "samples": dict(NORMALIZED_HISTOGRAM_REPRESENTATION)
        },
        "method": str(method),
        "training_config": dict(training_config),
        "training_output_dir": str(Path(output_dir).expanduser().resolve()),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "weights": "ema" if model_subfolder == "unet_ema" else "raw",
        "model_subfolder": str(model_subfolder),
        "sampling_config": dict(sampling_config),
        "inference_settings": dict(inference_settings),
        "source_seed": int(source_seed),
        "num_samples_per_condition": int(samples.shape[1]),
        "state_size": int(state_size),
        "deterministic": bool(deterministic),
        "reference_path": str(Path(primary_reference_path).expanduser().resolve()),
        "sampling_wall_seconds": float(sampling_wall_seconds),
        "sampler_diagnostic_totals": dict(sampler_diagnostic_totals or {}),
        "generated_unique_count": unique_counts,
        "maximum_mass_absolute_error": float(mass_error.max()),
    }

    return artifact


def save_or_reuse_sampling_artifact(
    path: str | Path,
    candidate: dict[str, Any],
    *,
    reuse_existing: bool,
    primary_reference: dict[str, Any],
    oracle_reference: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    destination = Path(path)
    if reuse_existing and destination.is_file():
        existing = load_torch_artifact(destination)
        if not isinstance(existing, dict):
            raise ValueError("Existing samples.pt is not a dictionary artifact.")
        # Complete provenance equality is intentional; elapsed time and diagnostics
        # belong to the cached generation and are therefore excluded from invocation identity.
        ignored = {
            "samples",
            "sampling_wall_seconds",
            "sampler_diagnostic_totals",
            "generated_unique_count",
            "maximum_mass_absolute_error",
        }
        for key in set(candidate) - ignored:
            if key not in existing or not _nested_equal(existing[key], candidate[key]):
                raise ValueError(
                    f"Cached samples.pt has stale provenance field {key!r}."
                )
        validate_evaluation_artifacts(
            existing, primary_reference, oracle_reference, sample_count=sample_count
        )
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(destination, candidate)
    return candidate


def load_reusable_sampling_artifact(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    primary_reference: dict[str, Any],
    oracle_reference: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    """Validate cache identity before any model forward pass is performed."""

    artifact = load_torch_artifact(path)
    if not isinstance(artifact, dict):
        raise ValueError("Existing samples.pt is not a dictionary artifact.")
    for key, expected in expected_provenance.items():
        if key not in artifact or not _nested_equal(artifact[key], expected):
            raise ValueError(f"Cached samples.pt has stale provenance field {key!r}.")
    validate_evaluation_artifacts(
        artifact, primary_reference, oracle_reference, sample_count=sample_count
    )
    return artifact


def evaluate_and_write_metrics(
    samples_artifact: dict[str, Any],
    primary_reference: dict[str, Any],
    oracle_reference: dict[str, Any],
    output_dir: str | Path,
    *,
    settings: EvaluationSettings,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, summary = evaluate_saved_samples_artifacts(
        samples_artifact,
        primary_reference,
        oracle_reference,
        settings=settings,
        device=device,
    )
    metrics = Path(output_dir) / "metrics"
    atomic_write_json(metrics / "evaluation_raw.json", raw)
    atomic_write_json(metrics / "evaluation_summary.json", summary)
    for score_name in ("energy_score", "kernel_score"):
        atomic_write_json(metrics / f"{score_name}.json", summary[score_name])
    atomic_write_json(
        metrics / "evaluation_settings.json", dataclasses.asdict(settings)
    )
    return raw, summary


__all__ = [
    "PosteriorConditionDataset",
    "SAMPLING_ARTIFACT_SCHEMA_VERSION",
    "build_sampling_artifact",
    "evaluate_and_write_metrics",
    "load_reusable_sampling_artifact",
    "load_and_validate_references",
    "parse_bool",
    "save_or_reuse_sampling_artifact",
]
