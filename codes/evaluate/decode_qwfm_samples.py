"""Apply a different QWFM decoder to saved quantiles, without network inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate.generate_proper_score_samples import validate_sample_cache
from utils.io_utils import load_torch_artifact
from utils.nmr_utils import decode_quantiles, make_normalized_log_rate_edges
from utils.predictive_generation_utils import _save_cache, _verified_cache, file_sha256
from utils.proper_score_conditions import load_proper_score_truth
from utils.proper_score_locks import model_generation_lock
from utils.quantile_decoder_artifacts import (
    decoder_comparison_metadata, reuse_test_conditions, validate_saved_quantiles,
)


def decode_artifact(source: dict, *, decoder: str = "linear_cdf", batch_size: int = 256) -> dict:
    """Preserve the exact generated state, changing only its histogram decoding."""
    if source.get("method") != "quantile_wasserstein":
        raise ValueError("Only QWFM samples can be re-decoded.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    q = validate_saved_quantiles(source)
    # Normalize in the original grid dtype exactly as the sampler does. The
    # decoder itself promotes arithmetic to float64 after fixing these edges.
    edges = make_normalized_log_rate_edges(source["grid"]["log_rate_edges"])
    flat = q.reshape(-1, q.shape[-1])
    batches = [decode_quantiles(part, edges, decoder=decoder).float()
               for part in flat.split(batch_size)]
    samples = torch.cat(batches).reshape(*q.shape[:2], len(edges) - 1)
    metadata = decoder_comparison_metadata(q, decoder)
    artifact = dict(source)
    artifact.update(
        samples=samples, decoder_comparison=metadata,
        maximum_mass_absolute_error=float((samples.sum(-1) - 1).abs().max()),
        generated_unique_count=torch.tensor([torch.unique(row, dim=0).shape[0] for row in samples]),
        sampling_config={**source["sampling_config"], "histogram_decoder": metadata["decoder_signature"]},
        proper_score_provenance={**source["proper_score_provenance"],
                                 "decoder_signature": metadata["decoder_signature"]},
    )
    return artifact


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source_run_dir", type=Path, required=True)
    parser.add_argument("--output_run_dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--decoder", choices=("linear_cdf",), default="linear_cdf")
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--test_dataset_seed", type=int, default=2026092901)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    args = parser.parse_args(argv)
    for name in ("num_conditions", "batch_size", "num_cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive.")
    source_run, output_run = args.source_run_dir.resolve(), args.output_run_dir.resolve()
    if (source_run == output_run or source_run in output_run.parents
            or output_run in source_run.parents):
        parser.error("Input and output runs must be separate sibling trees.")
    torch.set_num_threads(args.num_cpu_threads)
    truth = load_proper_score_truth(source_run, num_conditions=args.num_conditions,
                                   test_dataset_seed=args.test_dataset_seed)
    reuse_test_conditions(source_run, output_run)
    outputs = []
    for path in sorted(source_run.rglob("samples.pt")):
        # Select by embedded variant rather than the training directory name,
        # which may be customized. Another model can publish samples.pt before
        # its checksum sidecar without blocking this model's re-decoding.
        candidate = load_torch_artifact(path)
        if not isinstance(candidate, dict):
            raise ValueError(f"Expected a prediction artifact dictionary: {path}")
        if candidate.get("proper_score_provenance", {}).get("variant") != args.variant:
            del candidate
            continue
        del candidate
        source = _verified_cache(path, True)
        provenance = source.get("proper_score_provenance", {})
        if provenance.get("variant") != args.variant:
            raise ValueError(f"Prediction variant changed while selecting its cache: {path}")
        expected = dict(provenance)
        expected.pop("quantile_sha256", None)
        validate_sample_cache(source, truth, expected)
        validate_saved_quantiles(source)
        destination = output_run / path.relative_to(source_run)
        identity = {"samples_path": str(path), "samples_sha256": file_sha256(path),
                    "network_forward_performed": False}
        with model_generation_lock(output_run, str(path.relative_to(source_run).parts[0])):
            previous = _verified_cache(destination, True)
            if previous is not None:
                validate_saved_quantiles(previous)
                if (previous.get("decoder_source") != identity
                        or previous.get("decoder_comparison") != decoder_comparison_metadata(source["quantiles"], args.decoder)):
                    raise ValueError(f"Existing decoded cache belongs to another source/decoder: {destination}")
                target_expected = {**expected, "decoder_signature": previous["decoder_comparison"]["decoder_signature"]}
                validate_sample_cache(previous, truth, target_expected)
                print(f"reuse decoded samples: {destination}", flush=True)
            else:
                started = time.perf_counter()
                result = decode_artifact(source, decoder=args.decoder, batch_size=args.batch_size)
                result["decoder_source"] = identity
                result["decoding_wall_seconds"] = time.perf_counter() - started
                target_expected = dict(result["proper_score_provenance"])
                target_expected.pop("quantile_sha256")
                validate_sample_cache(result, truth, target_expected)
                _save_cache(destination, result)
                print(f"decoded without inference: {destination}", flush=True)
            outputs.append(destination)
    if not outputs:
        raise ValueError(f"No saved quantile samples for {args.variant} in {source_run}")
    return outputs


if __name__ == "__main__":
    main()
