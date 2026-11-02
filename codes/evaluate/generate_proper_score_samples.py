# ruff: noqa: E402
"""Generate learned-model samples from prepared independent ES/KS test pairs.

This production evaluation uses one chosen model and a shared test split. It is
independent of posterior references and diagnostic G/K sweeps. The saved samples
are evaluated offline by evaluate_proper_scores.py without another forward pass.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from utils.io_utils import atomic_write_json
from utils.model_variant_utils import FLOW_METHODS, VARIANTS, resolve_training_dir
from utils.posterior_evaluation_utils import (
    PosteriorConditionDataset, build_sampling_artifact, parse_bool,
)
from utils.predictive_generation_utils import (
    _collective, _derived_seed, _duration, _GenerationProgress, _main_action,
    _positive_list, _read_json, _save_cache, _verified_cache,
    effective_batch_size, file_sha256, PROPER_SCORE_PROTOCOL_VERSION, PROPER_SCORE_TEST_SPLIT,
)
from utils.predictive_model_utils import PredictiveModel, inspect_training_model
from utils.proper_score_conditions import load_proper_score_truth
from utils.proper_score_locks import model_generation_lock
from utils.quantile_decoder_artifacts import (
    decoder_comparison_metadata, validate_saved_quantiles,
)
from utils.nmr_utils import qwfm_decoder_signature


PROTOCOL = PROPER_SCORE_PROTOCOL_VERSION
TEST_SPLIT = PROPER_SCORE_TEST_SPLIT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--training_dir", type=Path)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--save_quantiles", type=parse_bool, default=False,
                        help="QWFM only: also save final quantiles for offline paired decoder comparisons.")
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--num_posterior_samples", type=int, default=8)
    parser.add_argument("--eval_ode_steps", type=_positive_list,
                        help="Comma-separated NFEs; default 256 for flow matching, 1 otherwise.")
    parser.add_argument("--checkpoint_path", default="latest")
    parser.add_argument("--model_subfolder", choices=("unet", "unet_ema"), default="unet_ema")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_generated_batch_size", type=int, default=512)
    parser.add_argument("--max_decode_batch_size", type=int, default=1024)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--test_dataset_seed", type=int, default=2026092901)
    parser.add_argument("--source_seed", type=int, default=2026092902)
    parser.add_argument("--reuse_existing", type=parse_bool, default=True,
                        help="Reuse matching predictions; shared test truths are always reused and validated.")
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--progress_interval_seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    for name in ("num_conditions", "num_posterior_samples", "eval_batch_size",
                 "max_generated_batch_size", "max_decode_batch_size", "num_cpu_threads",
                 "progress_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive.")
    for name in ("test_dataset_seed", "source_seed"):
        if not 0 <= getattr(args, name) < 2**63:
            parser.error(f"{name} must lie in [0, 2**63).")
    if args.dataloader_num_workers < 0:
        parser.error("dataloader_num_workers must be nonnegative.")
    if not math.isfinite(args.progress_interval_seconds) or args.progress_interval_seconds <= 0:
        parser.error("progress_interval_seconds must be finite and positive.")
    method = VARIANTS[args.variant][0]
    if args.save_quantiles and method != "quantile_wasserstein":
        parser.error("--save_quantiles is supported only for QWFM.")
    args.eval_ode_steps = args.eval_ode_steps or ((256,) if method in FLOW_METHODS else (1,))
    if method not in FLOW_METHODS and args.eval_ode_steps != (1,):
        parser.error("This model has no ODE integration; --eval_ode_steps must be 1.")
    if not method.startswith("deterministic_regression") and args.num_posterior_samples < 2:
        parser.error("Stochastic ES/KS requires at least two draws per condition.")
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    if args.training_dir is not None:
        args.training_dir = args.training_dir.expanduser().resolve()
    if args.run_dir is None:
        args.run_dir = (args.experiment_dir / "proper_scores"
                        / f"g{args.num_conditions}_k{args.num_posterior_samples}")
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "generation_manifest.json"
    result = _read_json(path) if path.exists() else {"protocol_version": PROTOCOL, "jobs": {}}
    if result.get("protocol_version") != PROTOCOL or not isinstance(result.get("jobs"), dict):
        raise ValueError("Existing manifest belongs to a different evaluation protocol.")
    return result


def _job_key(name: str, method: str, nfe: int) -> str:
    return f"{name}/nfe_{nfe:04d}" if method in FLOW_METHODS else name


def _load_truth(args: argparse.Namespace, accelerator: Accelerator, spec: Any) -> dict[str, Any]:
    """Read and validate the model-independent test split; never generate it."""
    truth = _collective(accelerator, lambda: load_proper_score_truth(
        args.run_dir, num_conditions=args.num_conditions,
        test_dataset_seed=args.test_dataset_seed, spec=spec,
    ), "load prepared test conditions")
    accelerator.print(f"test conditions: loaded {args.num_conditions} prepared truths (read only)", flush=True)
    return truth


def _as_reference(truth: dict[str, Any]) -> dict[str, Any]:
    return {**truth, "condition_ids": truth["sample_id"], "measurements": truth["measurement"],
            "truth_histograms": truth["histogram"]}


def validate_sample_cache(artifact: dict[str, Any], truth: dict[str, Any], expected: dict[str, Any]) -> None:
    provenance = dict(artifact.get("proper_score_provenance", {}))
    if expected.get("saved_quantiles"):
        validate_saved_quantiles(artifact)
        if provenance.pop("quantile_sha256", None) != artifact["quantile_sha256"]:
            raise ValueError("Quantile provenance is inconsistent.")
    if provenance != expected:
        raise ValueError("Cached predictions have different checkpoint, seed, or sampling settings.")
    if artifact.get("specification_hash") != truth["specification_hash"]:
        raise ValueError("Cached predictions use a different simulator.")
    if not torch.equal(artifact["condition_ids"], truth["sample_id"]) or not torch.equal(artifact["measurements"], truth["measurement"]):
        raise ValueError("Cached prediction conditions do not match the independent truth split.")
    samples = artifact["samples"]
    shape = (len(truth["sample_id"]), expected["stored_sample_count"], truth["histogram"].shape[-1])
    if samples.shape != shape or not torch.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("Cached prediction shape or values are invalid.")
    if not torch.allclose(samples.sum(-1), torch.ones(shape[:2]), atol=2e-4, rtol=0):
        raise ValueError("Cached predictions are not probability histograms.")


def _generate_job(args: argparse.Namespace, accelerator: Accelerator, name: str,
                  info: dict[str, Any], nfe: int, truth: dict[str, Any]) -> None:
    job_started = time.perf_counter()
    key = _job_key(name, info["method"], nfe)
    accelerator.print(f"{key}: checking prediction caches...", flush=True)
    path = args.run_dir / key / "samples.pt"
    count = len(truth["sample_id"])
    stored_k = 1 if info["deterministic"] else args.num_posterior_samples
    batch_size = effective_batch_size(count, accelerator.num_processes, args.eval_batch_size,
                                     stored_k, args.max_generated_batch_size)
    truth_hash = file_sha256(args.run_dir / "test_conditions.pt")
    seed = _derived_seed(args.source_seed, PROTOCOL, name, nfe, truth_hash)
    expected = {
        "protocol_version": PROTOCOL, "variant": args.variant, "conditions_sha256": truth_hash,
        "checkpoint_identity": info["checkpoint_identity"], "model_subfolder": args.model_subfolder,
        "model_name": name, "method": info["method"], "nfe": nfe,
        "stored_sample_count": stored_k, "requested_sample_count": args.num_posterior_samples,
        "master_source_seed": args.source_seed, "derived_source_seed": seed,
        "world_size": accelerator.num_processes, "eval_batch_size": batch_size,
        "mixed_precision": args.mixed_precision, "max_decode_batch_size": args.max_decode_batch_size,
    }
    if args.save_quantiles:
        expected.update(saved_quantiles=True, decoder_signature=qwfm_decoder_signature("hard_bin"))
    manifest = _manifest(args.run_dir)
    for field, value in (("num_conditions", count), ("requested_sample_count", args.num_posterior_samples),
                         ("conditions_sha256", truth_hash), ("specification_hash", info["spec"].specification_hash)):
        if field in manifest and manifest[field] != value:
            raise ValueError(f"Existing run manifest differs in {field}; choose a new run_dir.")
    artifact = _collective(accelerator, lambda: _verified_cache(path, args.reuse_existing), "prediction cache")
    if artifact is not None:
        _collective(accelerator, lambda: validate_sample_cache(artifact, truth, expected), "prediction identity")
        accelerator.print(f"reuse: {key} G={count} K={stored_k} (no new inference)", flush=True)
    else:
        accelerator.print(f"generate: {key} G={count} K={stored_k}, conditions/rank={count//accelerator.num_processes}, batch/rank={batch_size}", flush=True)
        batches, id_batches, quantile_batches = [], [], []
        start = time.perf_counter()
        accelerator.print(f"{key}: loading model...", flush=True)
        sampler = _collective(
            accelerator,
            lambda: PredictiveModel(info, accelerator, _derived_seed(seed, accelerator.process_index),
                                    max_decode_batch_size=args.max_decode_batch_size),
            f"load {key}",
        )
        dataset = PosteriorConditionDataset(_as_reference(truth))
        loader = accelerator.prepare(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=args.dataloader_num_workers))
        local_count = 0
        progress = _GenerationProgress(accelerator, f"{key} sampling", count,
                                       args.progress_every, args.progress_interval_seconds)
        for step, batch in enumerate(loader):
            sample_options = {"return_quantiles": True} if args.save_quantiles else {}
            with torch.inference_mode(), accelerator.autocast():
                result = _collective(
                    accelerator,
                    lambda: sampler.sample(batch["measurement"], num_samples=stored_k, nfe=nfe,
                                           **sample_options),
                    f"sample {key} batch {step}",
                )
            if args.save_quantiles:
                samples, quantiles = result
            else:
                samples = result
            local_count += len(samples)
            if args.save_quantiles:
                gathered, ids, gathered_quantiles = accelerator.gather_for_metrics(
                    (samples, batch["condition_id"], quantiles))
            else:
                gathered, ids = accelerator.gather_for_metrics((samples, batch["condition_id"]))
            if accelerator.is_main_process:
                batches.append(gathered.detach().cpu().float())
                id_batches.append(ids.detach().cpu())
                if args.save_quantiles:
                    quantile_batches.append(gathered_quantiles.detach().cpu().float())
            progress.update(local_count * accelerator.num_processes, step + 1)
        rank_counts = accelerator.gather(torch.tensor([local_count], device=accelerator.device)).cpu().tolist()
        if rank_counts != [count // accelerator.num_processes] * accelerator.num_processes:
            raise RuntimeError("Ranks did not generate exactly equal non-overlapping condition counts.")
        del sampler, loader
        accelerator.free_memory()
        gc.collect()
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        elapsed = float(accelerator.gather(torch.tensor([time.perf_counter()-start], device=accelerator.device)).max())

        def save() -> None:
            ids = torch.cat(id_batches)
            order = ids.argsort()
            if not torch.equal(ids[order], truth["sample_id"]):
                raise ValueError("Prediction generation lost, reordered, or duplicated conditions.")
            samples = torch.cat(batches)[order]
            artifact = build_sampling_artifact(
                spec=info["spec"], primary_reference=_as_reference(truth), samples=samples,
                method=info["method"], training_config=info["training_config"],
                output_dir=args.experiment_dir/name, checkpoint=info["checkpoint"],
                model_subfolder=args.model_subfolder, primary_reference_path=args.run_dir/"test_conditions.pt",
                state_size=info["state_size"], deterministic=info["deterministic"],
                sampling_config={"ode_steps": nfe} if info["method"] in FLOW_METHODS else {},
                source_seed=args.source_seed, sampling_wall_seconds=elapsed,
                inference_settings={"world_size": accelerator.num_processes, "eval_batch_size": batch_size,
                                    "mixed_precision": args.mixed_precision},
            )
            artifact.update(proper_score_provenance=expected, rank_condition_counts=rank_counts,
                            truth_source="independent_simulator_joint_pairs",
                            posterior_reference_samples_used=False,
                            requested_num_samples_per_condition=args.num_posterior_samples)
            if args.save_quantiles:
                quantiles = torch.cat(quantile_batches)[order]
                metadata = decoder_comparison_metadata(quantiles, "hard_bin")
                artifact.update(quantiles=quantiles, quantile_sha256=metadata["quantile_sha256"],
                                decoder_comparison=metadata)
                artifact["sampling_config"]["histogram_decoder"] = metadata["decoder_signature"]
                artifact["proper_score_provenance"] = {
                    **expected, "quantile_sha256": metadata["quantile_sha256"],
                }
            validate_sample_cache(artifact, truth, expected)
            _save_cache(path, artifact)

        accelerator.print(f"{key}: saving and checksumming (sampling ETA excludes this step)...", flush=True)
        _main_action(accelerator, save, f"save {key}")

    def record() -> None:
        # Each model/NFE owns this file. No common read-modify-write is needed.
        atomic_write_json(path.parent / "generation_manifest.json", {
            "protocol_version": PROTOCOL, "num_conditions": count,
            "requested_sample_count": args.num_posterior_samples,
            "conditions_file": str(args.run_dir / "test_conditions.pt"),
            "conditions_sha256": truth_hash,
            "specification_hash": info["spec"].specification_hash,
            "jobs": {key: {
                "variant": args.variant, "model_name": name, "method": info["method"], "nfe": nfe,
                "checkpoint": str(info["checkpoint"]), "checkpoint_identity": info["checkpoint_identity"],
                "samples_file": str(path.relative_to(args.run_dir)), "samples_sha256": file_sha256(path),
                "stored_sample_count": stored_k, "requested_sample_count": args.num_posterior_samples,
            }},
        })

    _main_action(accelerator, record, "save model generation manifest")
    accelerator.print(f"{key}: complete | elapsed={_duration(time.perf_counter()-job_started)}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_started = time.perf_counter()
    torch.set_num_threads(args.num_cpu_threads)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.num_conditions % accelerator.num_processes:
        raise ValueError("num_conditions must be divisible by the number of GPU processes.")
    accelerator.print(f"proper-score generation: {args.variant}"
                      f" | processes={accelerator.num_processes}", flush=True)
    with ExitStack() as locks:
        training_dir = _collective(
            accelerator, lambda: resolve_training_dir(args.experiment_dir, args.variant, args.training_dir),
            "model variant validation",
        )
        _main_action(accelerator, lambda: locks.enter_context(
            model_generation_lock(args.run_dir, training_dir.name)
        ), "lock selected model")
        info = _collective(
            accelerator, lambda: inspect_training_model(training_dir, args.checkpoint_path, args.model_subfolder),
            "checkpoint validation",
        )
        truth = _load_truth(args, accelerator, info["spec"])
        for nfe in args.eval_ode_steps:
            _generate_job(args, accelerator, training_dir.name, info, nfe, truth)
        accelerator.print(f"Saved proper-score samples: {args.run_dir}"
                          f" | total_elapsed={_duration(time.perf_counter()-run_started)}", flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
