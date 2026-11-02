"""Read and validate the immutable, model-independent ES/KS truth split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from data.spec import ContinuousNMRSpec
from utils.predictive_generation_utils import (
    PROPER_SCORE_PROTOCOL_VERSION,
    PROPER_SCORE_TEST_SPLIT,
    _verified_cache,
)


def validate_proper_score_truth(truth: dict[str, Any], expected: dict[str, Any]) -> None:
    """Reject changed simulation settings or incomplete/corrupt test pairs."""
    if truth.get("proper_score_provenance") != expected:
        raise ValueError("Existing truth cache has different proper-score settings.")
    count = expected["num_conditions"]
    ids = truth.get("sample_id")
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long
            or not torch.equal(ids, torch.arange(count))):
        raise ValueError("Truth condition IDs must be the complete sorted independent test split.")
    if (truth.get("specification_hash") != expected["specification_hash"]
            or truth.get("split") != PROPER_SCORE_TEST_SPLIT
            or truth.get("master_seed") != expected["master_seed"]):
        raise ValueError("Truth simulator specification or seed namespace differs.")
    spec = ContinuousNMRSpec.from_dict(truth["specification"])
    if spec.specification_hash != expected["specification_hash"]:
        raise ValueError("Truth simulator specification does not match its recorded hash.")
    for field, width in (("measurement", spec.num_measurements),
                         ("histogram", spec.num_spectrum_bins)):
        value = truth.get(field)
        if (not isinstance(value, torch.Tensor) or value.shape != (count, width)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid truth field: {field}")
    histogram = truth["histogram"]
    if (histogram < 0).any() or not torch.allclose(
        histogram.sum(-1), torch.ones(count, dtype=histogram.dtype), atol=2e-4, rtol=0,
    ):
        raise ValueError("Truth spectra must be probability histograms.")


def load_proper_score_truth(
    run_dir: Path, *, num_conditions: int, test_dataset_seed: int,
    spec: ContinuousNMRSpec | None = None,
) -> dict[str, Any]:
    """Load checked truths without creating files, locks, or directories."""
    path = Path(run_dir) / "test_conditions.pt"
    truth = _verified_cache(path, True)
    if truth is None:
        raise FileNotFoundError(
            f"Shared independent test conditions are missing: {path}. "
            "Run scripts/<experiment>/proper_scores/00_prepare_test_conditions.sh first "
            "with the same run directory, G and test_dataset_seed. "
            "Model generation only reads these prepared (y, h) pairs."
        )
    if spec is None:
        spec = ContinuousNMRSpec.from_dict(truth["specification"])
    expected = {
        "protocol_version": PROPER_SCORE_PROTOCOL_VERSION,
        "num_conditions": num_conditions,
        "specification_hash": spec.specification_hash,
        "master_seed": test_dataset_seed,
        "split": PROPER_SCORE_TEST_SPLIT,
    }
    validate_proper_score_truth(truth, expected)
    return truth
