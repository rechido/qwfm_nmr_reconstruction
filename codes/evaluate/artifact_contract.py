"""Shared representation contract for cached posterior-sample artifacts."""

from __future__ import annotations

from typing import Any

import torch


REPRESENTATION_METADATA_FIELD = "representation_metadata"
NORMALIZED_HISTOGRAM_REPRESENTATION = {
    "kind": "histogram_bin_mass",
    "normalization": "unit_mass",
    "role": "distribution_metrics_and_spectrum_visualization",
}
RAW_FORWARD_SPECTRUM_REPRESENTATION = {
    "kind": "spectrum_bin_amplitude",
    "normalization": "none",
    "role": "forward_model_diagnostics",
}


def classical_representation_metadata() -> dict[str, dict[str, str]]:
    """Return a fresh, JSON-safe description of the two classical outputs."""

    return {
        "samples": dict(NORMALIZED_HISTOGRAM_REPRESENTATION),
        "forward_spectra": dict(RAW_FORWARD_SPECTRUM_REPRESENTATION),
    }


def validate_optional_forward_spectra(
    artifact: dict[str, Any],
    *,
    groups: int,
    num_bins: int,
    artifact_name: str,
) -> torch.Tensor | None:
    """Validate and return an optional raw-amplitude forward representation.

    ``samples`` remains the unit-mass distribution representation.  An artifact
    that supplies ``forward_spectra`` must state the representation explicitly,
    so a normalized and a solver-native spectrum cannot be silently confused.
    The sample axis may have length one because deterministic inverse solvers
    produce one point reconstruction per condition.
    """

    forward_spectra = artifact.get("forward_spectra")
    if forward_spectra is None:
        return None
    if not isinstance(forward_spectra, torch.Tensor):
        raise ValueError(f"{artifact_name}.forward_spectra must be a tensor.")
    if forward_spectra.ndim != 3 or tuple(forward_spectra.shape[::2]) != (
        groups,
        num_bins,
    ):
        raise ValueError(
            f"{artifact_name}.forward_spectra must have shape [G,K_forward,N]."
        )
    if forward_spectra.shape[1] < 1:
        raise ValueError(
            f"{artifact_name}.forward_spectra must contain at least one spectrum."
        )
    if not forward_spectra.is_floating_point():
        raise ValueError(
            f"{artifact_name}.forward_spectra must use a floating-point dtype."
        )
    if not bool(torch.isfinite(forward_spectra).all()):
        raise ValueError(f"{artifact_name}.forward_spectra contains non-finite values.")
    if bool((forward_spectra < 0).any()):
        raise ValueError(f"{artifact_name}.forward_spectra contains negative values.")

    metadata = artifact.get(REPRESENTATION_METADATA_FIELD)
    if not isinstance(metadata, dict):
        raise ValueError(
            f"{artifact_name} supplies forward_spectra without explicit "
            f"{REPRESENTATION_METADATA_FIELD}."
        )
    if metadata.get("samples") != NORMALIZED_HISTOGRAM_REPRESENTATION:
        raise ValueError(
            f"{artifact_name} has invalid samples representation metadata."
        )
    if metadata.get("forward_spectra") != RAW_FORWARD_SPECTRUM_REPRESENTATION:
        raise ValueError(
            f"{artifact_name} has invalid forward_spectra representation metadata."
        )
    return forward_spectra


__all__ = [
    "NORMALIZED_HISTOGRAM_REPRESENTATION",
    "RAW_FORWARD_SPECTRUM_REPRESENTATION",
    "REPRESENTATION_METADATA_FIELD",
    "classical_representation_metadata",
    "validate_optional_forward_spectra",
]
