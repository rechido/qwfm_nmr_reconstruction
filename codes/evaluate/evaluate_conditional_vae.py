# ruff: noqa: E402
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))
from evaluate.evaluate_saved_samples import EvaluationSettings
from samplers.conditional_vae_sampler import ConditionalVAESampler
from utils.conditional_vae_utils import (
    CONDITIONAL_VAE_INPUT_REPRESENTATION,
    CONDITIONAL_VAE_OUTPUT_PARAMETERIZATION,
    CONDITIONAL_VAE_PRIOR,
    CONDITIONAL_VAE_TIMESTEP,
    get_conditional_vae_architecture_signature,
    load_conditional_vae_pretrained,
    validate_conditional_vae_model_config,
)
from utils.continuous_experiment_utils import (
    add_continuous_dataset_arguments,
    continuous_dataset_config_values,
    spec_from_args,
    validate_continuous_dataset_arguments,
)
from utils.checkpoint_utils import validate_checkpoint_path
from utils.nmr_utils import (
    find_latest_checkpoint,
    make_condition,
    save_reconstruction_figure,
    save_reconstruction_grid_figure,
)
from utils.posterior_evaluation_utils import (
    PosteriorConditionDataset,
    SAMPLING_ARTIFACT_SCHEMA_VERSION,
    build_sampling_artifact,
    evaluate_and_write_metrics,
    load_and_validate_references,
    load_reusable_sampling_artifact,
    parse_bool,
    save_or_reuse_sampling_artifact,
)

EXPERIMENT_NAME = "conditional_vae"


def _checkpoint_iteration(state: dict[str, Any]) -> int:
    value = state.get("global_iteration")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Checkpoint global_iteration must be nonnegative.")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the conditional VAE on SMC posterior conditions."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_subfolder", choices=("unet", "unet_ema"), required=True)
    parser.add_argument("--eval_output_subdir", type=str, required=True)
    parser.add_argument("--primary_reference", type=str, required=True)
    parser.add_argument("--oracle_reference", type=str, required=True)
    add_continuous_dataset_arguments(parser)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--encoder_hidden_dim", type=int, required=True)
    parser.add_argument("--encoder_num_layers", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, required=True)
    parser.add_argument("--num_posterior_samples", type=int, required=True)
    parser.add_argument("--num_visual_samples", type=int, required=True)
    parser.add_argument("--max_decode_batch_size", type=int, required=True)
    parser.add_argument("--reference_repeats", type=int, required=True)
    parser.add_argument("--nearest_k", type=int, required=True)
    parser.add_argument("--w2_num_quantiles", type=int, required=True)
    parser.add_argument("--distance_chunk_size", type=int, required=True)
    parser.add_argument("--metric_seed", type=int, required=True)
    parser.add_argument("--credible_interval_level", type=float, required=True)
    parser.add_argument("--kernel_bandwidth", type=float, default=None,
                        help="Fixed Gaussian bandwidth; default: independent simulator calibration median.")
    parser.add_argument("--reuse_existing", type=parse_bool, required=True)
    parser.add_argument("--peak_relative_threshold", type=float, required=True)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample_seed", type=int, required=True)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    bandwidth = getattr(args, "kernel_bandwidth", None)
    if bandwidth is not None and (not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("kernel_bandwidth must be finite and positive.")
    validate_continuous_dataset_arguments(args)
    for name in (
        "eval_batch_size",
        "num_posterior_samples",
        "num_visual_samples",
        "reference_repeats",
        "nearest_k",
        "w2_num_quantiles",
        "distance_chunk_size",
        "latent_dim",
        "encoder_hidden_dim",
        "encoder_num_layers",
        "max_decode_batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if (
        args.dataloader_num_workers < 0
        or args.num_visual_samples > args.num_posterior_samples
        or args.num_posterior_samples <= args.nearest_k
    ):
        raise ValueError("Invalid evaluation counts.")
    for name in ("seed", "sample_seed", "metric_seed"):
        if not 0 <= int(getattr(args, name)) < 2**63:
            raise ValueError(f"{name} must lie in [0,2**63).")
    if not 0.0 <= args.peak_relative_threshold <= 1.0:
        raise ValueError("peak_relative_threshold must lie in [0,1].")
    if not 0.0 < args.credible_interval_level < 1.0:
        raise ValueError("credible_interval_level must lie in (0,1).")
    relative = Path(args.eval_output_subdir)
    if (
        not args.output_dir.strip()
        or not args.checkpoint_path.strip()
        or not args.eval_output_subdir.strip()
        or relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
    ):
        raise ValueError("Output/checkpoint/evaluation paths are invalid.")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_path.strip().lower() == "latest":
        checkpoint = find_latest_checkpoint(
            Path(args.output_dir),
            require_ema=args.model_subfolder == "unet_ema",
        )
    else:
        requested = Path(args.checkpoint_path).expanduser()
        if not requested.is_absolute():
            requested = Path.cwd() / requested
        checkpoint = validate_checkpoint_path(
            requested,
            require_ema=args.model_subfolder == "unet_ema",
            allow_legacy_numbered=False,
        )
    if not (checkpoint / args.model_subfolder / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete conditional-VAE checkpoint: {checkpoint}")
    return checkpoint


def _matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
        )
    return actual == expected


def _validate_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _read_json_object(Path(args.output_dir) / "training_config.json")
    state = _read_json_object(checkpoint / "training_state.json")
    expected = {
        "experiment": EXPERIMENT_NAME,
        **continuous_dataset_config_values(args),
        "network_architecture": get_conditional_vae_architecture_signature(
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
            latent_dim=args.latent_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_num_layers=args.encoder_num_layers,
        ),
        "conditional_vae_input_representation": CONDITIONAL_VAE_INPUT_REPRESENTATION,
        "conditional_vae_output_parameterization": CONDITIONAL_VAE_OUTPUT_PARAMETERIZATION,
        "conditional_vae_prior": CONDITIONAL_VAE_PRIOR,
        "conditional_vae_timestep": CONDITIONAL_VAE_TIMESTEP,
        "reconstruction_loss": "mean_squared_histogram_error_v1",
        "kl_reduction": "mean_over_latent_coordinates_v1",
    }
    for label, config in (("root training_config", root), ("checkpoint training_config", state.get("training_config"))):
        if not isinstance(config, dict):
            raise ValueError(f"{label} is missing.")
        mismatches = [
            f"{name}: saved={config.get(name)!r}, evaluation={value!r}"
            for name, value in expected.items()
            if not _matches(config.get(name), value)
        ]
        if mismatches:
            raise ValueError(f"{label} mismatch:\n" + "\n".join(mismatches))
    if state.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint experiment mismatch.")
    _checkpoint_iteration(state)
    return root, state


def collective_raise_if_failed(
    accelerator: Accelerator,
    local_error: Exception | None,
    phase: str,
) -> None:
    failure = torch.tensor(
        int(local_error is not None), device=accelerator.device, dtype=torch.int64
    )
    failures = int(accelerator.reduce(failure, reduction="sum").item())
    if failures:
        if local_error is None:
            raise RuntimeError(f"{phase} failed on {failures} other process(es).")
        raise RuntimeError(f"{phase} failed: {local_error}") from local_error


def _expected_cache_provenance(
    *,
    args: Any,
    spec: Any,
    checkpoint: Path,
    training_config: dict[str, Any],
    sampling_config: dict[str, Any],
    inference_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": SAMPLING_ARTIFACT_SCHEMA_VERSION,
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "method": EXPERIMENT_NAME,
        "training_config": dict(training_config),
        "training_output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "weights": "ema" if args.model_subfolder == "unet_ema" else "raw",
        "model_subfolder": args.model_subfolder,
        "sampling_config": dict(sampling_config),
        "inference_settings": dict(inference_settings),
        "source_seed": int(args.sample_seed),
        "num_samples_per_condition": int(args.num_posterior_samples),
        "state_size": int(args.num_spectrum_bins),
        "deterministic": False,
        "reference_path": str(Path(args.primary_reference).expanduser().resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = spec_from_args(args)
    checkpoint = _resolve_checkpoint(args)
    training_config, checkpoint_state = _validate_checkpoint(args, checkpoint)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed, device_specific=True)
    if not 0 <= int(args.sample_seed) < 2**63 - accelerator.num_processes:
        raise ValueError("source seed plus process rank must lie in [0,2**63).")
    primary, oracle = load_and_validate_references(
        args.primary_reference, args.oracle_reference, spec
    )
    count = int(primary["condition_ids"].shape[0])
    global_batch = args.eval_batch_size * accelerator.num_processes
    minimum_count = 16
    if count < minimum_count or count % global_batch:
        raise ValueError(
            f"Reference condition count must be at least {minimum_count} and divisible by "
            "eval_batch_size times world size."
        )
    relative_evaluation = Path(args.eval_output_subdir)
    if (
        not args.eval_output_subdir.strip()
        or relative_evaluation.is_absolute()
        or relative_evaluation == Path(".")
        or ".." in relative_evaluation.parts
    ):
        raise ValueError("eval_output_subdir must be a safe nonempty relative path.")
    evaluation_dir = Path(args.output_dir) / relative_evaluation
    samples_path = evaluation_dir / "samples.pt"
    inference_settings = {
        "mixed_precision": args.mixed_precision,
        "world_size": accelerator.num_processes,
        "eval_batch_size": args.eval_batch_size,
    }
    sampling_config = {
        "latent_dim": args.latent_dim,
        "max_decode_batch_size": args.max_decode_batch_size,
        "sample_seed": args.sample_seed,
    }
    provenance = _expected_cache_provenance(
        args=args,
        spec=spec,
        checkpoint=checkpoint,
        training_config=training_config,
        sampling_config=sampling_config,
        inference_settings=inference_settings,
    )
    artifact: dict[str, Any] | None = None
    reuse_error: Exception | None = None
    if args.reuse_existing:
        try:
            artifact = load_reusable_sampling_artifact(
                samples_path,
                expected_provenance=provenance,
                primary_reference=primary,
                oracle_reference=oracle,
                sample_count=args.num_posterior_samples,
            )
        except Exception as error:
            reuse_error = error
    collective_raise_if_failed(accelerator, reuse_error, "cached-sample validation")

    if artifact is None:
        model = load_conditional_vae_pretrained(checkpoint / args.model_subfolder)
        validate_conditional_vae_model_config(
            model,
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
            latent_dim=args.latent_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_num_layers=args.encoder_num_layers,
        )
        reference_dataset = PosteriorConditionDataset(primary)
        model, loader = accelerator.prepare(
            model,
            DataLoader(
                reference_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.dataloader_num_workers,
                drop_last=False,
            ),
        )
        model.eval()
        evaluation_model = accelerator.unwrap_model(model)
        sampler = ConditionalVAESampler(
            evaluation_model,
            max_decode_batch_size=args.max_decode_batch_size,
        )
        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.sample_seed + accelerator.process_index
        )
        gathered_indices: list[torch.Tensor] = []
        gathered_samples: list[torch.Tensor] = []
        diagnostic_totals: dict[str, float] = {}
        accelerator.wait_for_everyone()
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        started = time.perf_counter()
        for batch in loader:
            measurements = batch["measurement"]
            measurements = measurements.to(accelerator.device)
            condition = make_condition(
                measurements,
                primary["grid"]["echo_times"],
                output_size=args.num_spectrum_bins,
                noise_standard_deviation=args.noise_standard_deviation,
            )
            with accelerator.autocast():
                samples, diagnostics = sampler.sample(
                    condition,
                    num_samples_per_condition=args.num_posterior_samples,
                    generator=generator,
                )
            all_indices, all_samples = accelerator.gather_for_metrics(
                (batch["condition_id"], samples)
            )
            if accelerator.is_main_process:
                gathered_indices.append(all_indices.cpu())
                gathered_samples.append(all_samples.cpu())
            for name, value in diagnostics.items():
                reduced = accelerator.reduce(value.to(torch.float64), reduction="sum")
                if accelerator.is_main_process:
                    diagnostic_totals[name] = diagnostic_totals.get(name, 0.0) + float(
                        reduced
                    )
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        local_elapsed = torch.tensor(
            [time.perf_counter() - started],
            device=accelerator.device,
            dtype=torch.float64,
        )
        elapsed = float(accelerator.gather(local_elapsed).max().item())
        post_error: Exception | None = None
        if accelerator.is_main_process:
            try:
                indices = torch.cat(gathered_indices)
                samples = torch.cat(gathered_samples)
                expected_ids = primary["condition_ids"].cpu()
                positions = {
                    int(condition_id): index
                    for index, condition_id in enumerate(expected_ids.tolist())
                }
                if set(indices.tolist()) != set(expected_ids.tolist()):
                    raise RuntimeError(
                        "Distributed evaluation did not cover each condition once."
                    )
                order = torch.tensor(
                    sorted(range(count), key=lambda index: positions[int(indices[index])]),
                    dtype=torch.int64,
                )
                samples = samples[order]
                artifact = build_sampling_artifact(
                    spec=spec,
                    primary_reference=primary,
                    samples=samples,
                    method=EXPERIMENT_NAME,
                    training_config=training_config,
                    output_dir=args.output_dir,
                    checkpoint=checkpoint,
                    model_subfolder=args.model_subfolder,
                    primary_reference_path=args.primary_reference,
                    state_size=args.num_spectrum_bins,
                    deterministic=False,
                    sampling_config=sampling_config,
                    source_seed=args.sample_seed,
                    sampling_wall_seconds=elapsed,
                    inference_settings=inference_settings,
                    sampler_diagnostic_totals=diagnostic_totals,
                )
                artifact = save_or_reuse_sampling_artifact(
                    samples_path,
                    artifact,
                    reuse_existing=False,
                    primary_reference=primary,
                    oracle_reference=oracle,
                    sample_count=args.num_posterior_samples,
                )
            except Exception as error:
                post_error = error
        collective_raise_if_failed(accelerator, post_error, "sample postprocessing")

    metric_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            assert artifact is not None
            evaluate_and_write_metrics(
                artifact,
                primary,
                oracle,
                evaluation_dir,
                settings=EvaluationSettings(
                    sample_count=args.num_posterior_samples,
                    reference_repeats=args.reference_repeats,
                    nearest_k=args.nearest_k,
                    w2_num_quantiles=args.w2_num_quantiles,
                    distance_chunk_size=args.distance_chunk_size,
                    seed=args.metric_seed,
                    credible_interval_level=args.credible_interval_level,
                    kernel_bandwidth=getattr(args, "kernel_bandwidth", None),
                ),
                device="cpu",
            )
            grid = primary["grid"]
            save_reconstruction_figure(
                evaluation_dir / "reconstruction.png",
                artifact["samples"][0, : args.num_visual_samples],
                primary["truth_histograms"][0],
                primary["measurements"][0],
                grid["rate_centers"],
                grid["echo_times"],
                grid["kernel"],
                args.noise_standard_deviation,
                figure_title=(
                    f"Conditional VAE | checkpoint iteration "
                    f"{_checkpoint_iteration(checkpoint_state)}"
                ),
            )
            save_reconstruction_grid_figure(
                evaluation_dir / "reconstruction_grid.png",
                artifact["samples"][:16],
                primary["truth_histograms"][:16],
                grid["rate_centers"],
            )
        except Exception as error:
            metric_error = error
    collective_raise_if_failed(
        accelerator, metric_error, "metric and figure postprocessing"
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
