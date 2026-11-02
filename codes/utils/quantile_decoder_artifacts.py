"""Immutable artifacts for paired QWFM decoder comparisons.

Training/checkpoint contracts remain unchanged. Decoder metadata describes only
the postprocessing of the saved final quantile state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from utils.nmr_utils import qwfm_decoder_signature
from utils.predictive_generation_utils import _verified_cache, file_sha256
from utils.proper_score_locks import file_lock


DECODER_COMPARISON_PROTOCOL = "qwfm_decoder_comparison_v1"


def quantile_tensor_sha256(quantiles: torch.Tensor) -> str:
    values = quantiles.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(values.shape), "dtype": str(values.dtype)},
                             sort_keys=True).encode("utf-8"))
    digest.update(values.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def decoder_comparison_metadata(quantiles: torch.Tensor, decoder: str) -> dict:
    return {
        "protocol_version": DECODER_COMPARISON_PROTOCOL,
        "decoder_name": decoder,
        "decoder_signature": qwfm_decoder_signature(decoder),
        "quantile_sha256": quantile_tensor_sha256(quantiles),
    }


def reuse_test_conditions(source_run: Path, destination_run: Path) -> None:
    """Link the existing, checksummed test split; never simulate or overwrite it."""
    source_run, destination_run = source_run.resolve(), destination_run.resolve()
    source = source_run / "test_conditions.pt"
    if _verified_cache(source, True) is None:
        raise FileNotFoundError(f"Existing test conditions are required: {source}")
    if source_run == destination_run:
        return
    with file_lock(destination_run / ".decoder_input.lock", blocking=True):
        for filename in ("test_conditions.pt", "test_conditions.sha256.json"):
            original = source_run / filename
            target = destination_run / filename
            if target.exists() or target.is_symlink():
                if not target.is_file() or file_sha256(target) != file_sha256(original):
                    raise ValueError(f"Different test input already exists: {target}")
            else:
                target.symlink_to(original)
        _verified_cache(destination_run / "test_conditions.pt", True)


def validate_saved_quantiles(artifact: dict) -> torch.Tensor:
    """Validate raw states and their content identity before any re-decoding."""
    quantiles = artifact.get("quantiles")
    samples = artifact["samples"]
    if (not isinstance(quantiles, torch.Tensor) or quantiles.ndim != 3
            or quantiles.shape[:2] != samples.shape[:2]
            or quantiles.shape[-1] != artifact["state_size"]
            or not quantiles.is_floating_point() or not torch.isfinite(quantiles).all()
            or (quantiles < 0).any() or (quantiles > 1).any()
            or (quantiles.diff(dim=-1) < 0).any()):
        raise ValueError("Missing or invalid saved final quantiles; regenerate QWFM with save_quantiles=true.")
    expected = quantile_tensor_sha256(quantiles)
    if artifact.get("quantile_sha256") != expected:
        raise ValueError("Saved quantile content differs from its recorded identity.")
    metadata = artifact.get("decoder_comparison", {})
    if metadata != decoder_comparison_metadata(quantiles, metadata.get("decoder_name", "")):
        raise ValueError("Invalid decoder comparison metadata.")
    signature = metadata["decoder_signature"]
    if artifact.get("sampling_config", {}).get("histogram_decoder") != signature:
        raise ValueError("Sampling settings and decoder metadata disagree.")
    if "proper_score_provenance" in artifact:
        provenance = artifact["proper_score_provenance"]
        if (provenance.get("decoder_signature") != signature
                or provenance.get("quantile_sha256") != expected
                or provenance.get("saved_quantiles") is not True):
            raise ValueError("Proper-score provenance and saved quantiles disagree.")
    return quantiles
