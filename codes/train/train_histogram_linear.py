# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import UNet1DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Subset


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from samplers.histogram_linear_sampler import HistogramLinearSampler
from utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    resolve_checkpoint,
    save_latest_accelerator_checkpoint,
    validate_checkpoint_path,
)
from trainers.histogram_linear_trainer import HistogramLinearTrainer
from utils.nmr_utils import (
    evaluate_histogram_samples,
    make_condition,
    save_reconstruction_figure,
    save_reconstruction_grid_figure,
    write_json,
)
from utils.continuous_experiment_utils import (
    add_continuous_dataset_arguments,
    continuous_dataset_config_values,
    json_safe_dataset_metadata,
    make_continuous_dataset_family,
    make_continuous_training_stream,
    validate_continuous_dataset_arguments,
)
from utils.unet_config import (
    get_unet_config,
    get_network_architecture_signature,
    unet_sample,
)
from utils.training_utils import (
    FixedFMProbeDataset,
    append_jsonl,
    build_fixed_fm_probe_bank,
    collect_euler_samples,
    format_console_log_record,
    make_fm_generalization_record,
    nested_config_values_match,
    reduce_fixed_fm_probe_statistics,
    require_fresh_training_history,
    restore_latest_fixed_evaluation_metrics,
    save_fm_generalization_curves,
    save_training_curves,
    slowest_distributed_elapsed_seconds,
    start_distributed_wall_timer,
    truncate_jsonl_after_iteration,
)


EXPERIMENT_NAME = "histogram_linear"
LEGACY_CONFIG_ALIASES = {
    "eval_ode_steps": ("eval_sample_steps",),
    "gradient_accumulation_batches": ("gradient_accumulation_steps",),
    "model_time_scale": ("fm_num_timesteps",),
}


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, 1/0, or yes/no.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train histogram-linear NMR flow matching."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset_seed", type=int, required=True)
    add_continuous_dataset_arguments(parser)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--num_train_iterations", type=int, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, required=True)
    parser.add_argument("--gradient_accumulation_batches", type=int, required=True)
    parser.add_argument("--log_every_iterations", type=int, required=True)
    parser.add_argument("--eval_every_iterations", type=int, required=True)
    parser.add_argument("--num_fixed_eval_samples", type=int, required=True)
    parser.add_argument("--fixed_eval_dataset_seed", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, required=True)
    parser.add_argument("--eval_dataloader_num_workers", type=int, required=True)
    parser.add_argument("--eval_samples_per_condition", type=int, required=True)
    parser.add_argument("--eval_ode_steps", type=int, required=True)
    parser.add_argument("--eval_num_visual_samples", type=int, required=True)
    parser.add_argument("--eval_source_seed", type=int, required=True)
    parser.add_argument("--eval_metric_num_quantiles", type=int, required=True)
    parser.add_argument("--eval_peak_relative_threshold", type=float, required=True)
    parser.add_argument(
        "--mixed_precision", choices=("no", "fp16", "bf16"), required=True
    )
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument(
        "--lr_scheduler",
        choices=(
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ),
        required=True,
    )
    parser.add_argument("--lr_warmup_iterations", type=int, required=True)
    parser.add_argument("--lr_num_cycles", type=int, required=True)
    parser.add_argument("--lr_power", type=float, required=True)
    parser.add_argument("--adam_beta1", type=float, required=True)
    parser.add_argument("--adam_beta2", type=float, required=True)
    parser.add_argument("--adam_weight_decay", type=float, required=True)
    parser.add_argument("--adam_epsilon", type=float, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument(
        "--use_ema", nargs="?", const=True, type=_parse_bool, required=True
    )
    parser.add_argument("--ema_decay", type=float, required=True)
    parser.add_argument("--ema_min_decay", type=float, required=True)
    parser.add_argument("--ema_update_after_iterations", type=int, required=True)
    parser.add_argument(
        "--ema_use_warmup", nargs="?", const=True, type=_parse_bool, required=True
    )
    parser.add_argument("--ema_inv_gamma", type=float, required=True)
    parser.add_argument("--ema_power", type=float, required=True)
    parser.add_argument("--checkpoint_every_iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source_seed", type=int, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, required=True)
    parser.add_argument("--model_time_scale", type=int, required=True)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_measurements",
        "num_spectrum_bins",
        "train_batch_size",
        "num_train_iterations",
        "log_every_iterations",
        "eval_every_iterations",
        "num_fixed_eval_samples",
        "eval_batch_size",
        "eval_samples_per_condition",
        "eval_ode_steps",
        "eval_num_visual_samples",
        "eval_metric_num_quantiles",
        "gradient_accumulation_batches",
        "checkpoint_every_iterations",
        "model_time_scale",
        "lr_num_cycles",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if (
        args.dataloader_num_workers < 0
        or args.eval_dataloader_num_workers < 0
        or args.lr_warmup_iterations < 0
    ):
        raise ValueError("Worker and warmup-iteration counts must be nonnegative.")
    for name in (
        "train_dataset_seed",
        "fixed_eval_dataset_seed",
        "seed",
        "source_seed",
        "eval_source_seed",
    ):
        if not 0 <= int(getattr(args, name)) < 2**63:
            raise ValueError(f"{name} must lie in [0, 2**63).")
    if args.num_spectrum_bins < 8:
        raise ValueError("num_spectrum_bins must be at least eight.")
    if not 0.0 < args.rate_minimum < args.rate_maximum:
        raise ValueError("Require 0 < rate_minimum < rate_maximum.")
    if not 0.0 < args.echo_time_minimum < args.echo_time_maximum:
        raise ValueError("Require 0 < echo_time_minimum < echo_time_maximum.")
    if args.noise_standard_deviation <= 0.0:
        raise ValueError("noise_standard_deviation must be positive.")
    validate_continuous_dataset_arguments(args)
    if args.learning_rate <= 0.0 or args.adam_epsilon <= 0.0:
        raise ValueError("learning_rate and adam_epsilon must be positive.")
    if args.eval_num_visual_samples > args.eval_samples_per_condition:
        raise ValueError(
            "eval_num_visual_samples cannot exceed eval_samples_per_condition."
        )
    if args.num_fixed_eval_samples < 16:
        raise ValueError(
            "num_fixed_eval_samples must be at least 16 for the 4x4 "
            "reconstruction grid."
        )
    if (
        not math.isfinite(args.eval_peak_relative_threshold)
        or not 0.0 <= args.eval_peak_relative_threshold <= 1.0
    ):
        raise ValueError("eval_peak_relative_threshold must lie in [0, 1].")
    if not math.isfinite(args.lr_power) or args.lr_power <= 0.0:
        raise ValueError("lr_power must be finite and positive.")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must be in [0, 1).")
    if args.adam_weight_decay < 0.0 or args.max_grad_norm <= 0.0:
        raise ValueError("Weight decay must be nonnegative and max_grad_norm positive.")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must lie in (0, 1).")
    if (
        not math.isfinite(args.ema_min_decay)
        or not 0.0 <= args.ema_min_decay <= args.ema_decay
    ):
        raise ValueError("ema_min_decay must lie in [0, ema_decay].")
    if args.ema_update_after_iterations < 0:
        raise ValueError("ema_update_after_iterations must be nonnegative.")
    if not math.isfinite(args.ema_inv_gamma) or args.ema_inv_gamma <= 0.0:
        raise ValueError("ema_inv_gamma must be finite and positive.")
    if not math.isfinite(args.ema_power) or args.ema_power <= 0.0:
        raise ValueError("ema_power must be finite and positive.")


def _checkpoint_artifacts_exist(checkpoint_root: Path) -> bool:
    return (
        (checkpoint_root / "latest").exists()
        or (checkpoint_root / "latest").is_symlink()
        or any(checkpoint_root.glob(".generation-*"))
        or any(checkpoint_root.glob(".staging-*"))
        or any(checkpoint_root.glob("checkpoint-*"))
    )


def _resolve_resume(args: argparse.Namespace) -> Path | None:
    requested = args.resume_from_checkpoint.strip()
    checkpoint_root = Path(args.output_dir) / "checkpoints"
    if requested.lower() == "none":
        if checkpoint_root.exists() and _checkpoint_artifacts_exist(checkpoint_root):
            raise FileExistsError(
                "Checkpoints exist; select latest/path or a new output_dir."
            )
        return None
    if requested.lower() == "latest":
        try:
            return resolve_checkpoint(
                args.output_dir,
                require_ema=args.use_ema,
                allow_legacy_numbered=False,
            )
        except FileNotFoundError as error:
            if checkpoint_root.exists() and _checkpoint_artifacts_exist(
                checkpoint_root
            ):
                raise RuntimeError(
                    "Checkpoint artifacts exist, but none is a complete resumable state."
                ) from error
            return None
    return validate_checkpoint_path(
        requested,
        require_ema=args.use_ema,
        allow_legacy_numbered=False,
    )


def _read_state(checkpoint: Path) -> dict[str, object]:
    path = checkpoint / "training_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return state


def _validate_resume(current: dict[str, object], state: dict[str, object]) -> None:
    saved = state.get("training_config")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint lacks a complete training_config.")
    mutable = {
        "output_dir",
        "num_train_iterations",
        "dataloader_num_workers",
        "log_every_iterations",
        "eval_every_iterations",
        "eval_batch_size",
        "eval_dataloader_num_workers",
        "eval_samples_per_condition",
        "eval_ode_steps",
        "eval_num_visual_samples",
        "eval_metric_num_quantiles",
        "eval_peak_relative_threshold",
        "checkpoint_every_iterations",
        "resume_from_checkpoint",
        "total_training_pairs",
    }
    mismatches: list[str] = []
    for key in sorted(set(current) - mutable):
        aliases = LEGACY_CONFIG_ALIASES.get(key, ())
        left = current[key]
        candidates = [
            (candidate_key, saved[candidate_key])
            for candidate_key in (key, *aliases)
            if candidate_key in saved
        ]
        if not candidates:
            if key == "network_architecture":
                mismatches.append(
                    "network_architecture: missing from checkpoint; "
                    "cannot verify the stock UNet1DModel architecture"
                )
                continue
            mismatches.append(f"{key}: missing from checkpoint")
            continue
        right = candidates[0][1]
        if any(
            not nested_config_values_match(value, right) for _, value in candidates[1:]
        ):
            names = ", ".join(name for name, _ in candidates)
            mismatches.append(f"{key}: conflicting checkpoint aliases ({names})")
            continue
        if not nested_config_values_match(left, right):
            mismatches.append(f"{key}: checkpoint={right!r}, current={left!r}")
    if mismatches:
        raise ValueError("Incompatible resume configuration:\n" + "\n".join(mismatches))


def _resume_progress(
    state: dict[str, object],
    *,
    requested_iterations: int,
) -> int:
    if state.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Only iteration-native checkpoint schema "
            f"v{CHECKPOINT_SCHEMA_VERSION} can resume this training stream."
        )
    if state.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            f"Checkpoint experiment mismatch: {state.get('experiment')!r}."
        )
    global_iteration = state.get("global_iteration")
    if (
        isinstance(global_iteration, bool)
        or not isinstance(global_iteration, int)
        or global_iteration < 0
    ):
        raise ValueError("Checkpoint global_iteration must be a nonnegative integer.")

    if global_iteration > requested_iterations:
        raise ValueError(
            "Checkpoint iteration exceeds the requested training horizon: "
            f"checkpoint={global_iteration}, requested={requested_iterations}."
        )
    return global_iteration


def _collective_raise_if_failed(
    accelerator: Accelerator,
    local_error: Exception | None,
    phase: str,
) -> None:
    failure = torch.tensor(
        1 if local_error is not None else 0,
        device=accelerator.device,
        dtype=torch.int64,
    )
    failure_count = int(accelerator.reduce(failure, reduction="sum").item())
    if failure_count:
        if local_error is None:
            raise RuntimeError(f"{phase} failed on {failure_count} other process(es).")
        raise RuntimeError(f"{phase} failed: {local_error}") from local_error


def _safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0.0:
        raise RuntimeError(f"Cannot compute {label}: denominator is nonpositive.")
    return numerator / denominator


@torch.no_grad()
def _collect_fixed_fm_probe_metrics(
    args: argparse.Namespace,
    accelerator: Accelerator,
    model,
    loader,
    dataset,
) -> dict[str, float | int]:
    error_squared_sum = torch.zeros(
        args.num_spectrum_bins,
        device=accelerator.device,
        dtype=torch.float64,
    )
    velocity_sum = torch.zeros_like(error_squared_sum)
    velocity_squared_sum = torch.zeros_like(error_squared_sum)
    sample_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    for targets, measurements, source_histograms, flow_times in loader:
        source_histograms = source_histograms.to(
            device=targets.device, dtype=targets.dtype
        )
        flow_times = flow_times.to(device=targets.device, dtype=targets.dtype)
        broadcast_time = flow_times[:, None]
        states = (1.0 - broadcast_time) * source_histograms + broadcast_time * targets
        target_velocity = targets - source_histograms
        condition = make_condition(
            measurements,
            dataset.echo_times,
            output_size=args.num_spectrum_bins,
            noise_standard_deviation=args.noise_standard_deviation,
        )
        model_input = torch.cat((states[:, None, :], condition), dim=1)
        predicted_velocity = unet_sample(
            model,
            model_input,
            flow_times * float(args.model_time_scale),
        )[:, 0, :]

        target_velocity_64 = target_velocity.to(torch.float64)
        prediction_error_64 = predicted_velocity.to(torch.float64) - target_velocity_64
        error_squared_sum += prediction_error_64.square().sum(dim=0)
        velocity_sum += target_velocity_64.sum(dim=0)
        velocity_squared_sum += target_velocity_64.square().sum(dim=0)
        sample_count += targets.shape[0]

    return reduce_fixed_fm_probe_statistics(
        accelerator,
        error_squared_sum,
        velocity_sum,
        velocity_squared_sum,
        sample_count,
    )


@torch.no_grad()
def _run_fixed_fm_generalization_evaluation(
    args: argparse.Namespace,
    accelerator: Accelerator,
    model,
    train_probe_loader,
    validation_probe_loader,
    dataset,
    output_dir: Path,
    iteration: int,
) -> None:
    was_training = model.training
    model.eval()
    try:
        train_metrics = _collect_fixed_fm_probe_metrics(
            args, accelerator, model, train_probe_loader, dataset
        )
        validation_metrics = _collect_fixed_fm_probe_metrics(
            args, accelerator, model, validation_probe_loader, dataset
        )
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    postprocess_error = None
    if accelerator.is_main_process:
        try:
            if train_metrics["sample_count"] != args.num_fixed_eval_samples:
                raise RuntimeError(
                    "Gathered train FM probe sample count mismatch: "
                    f"expected {args.num_fixed_eval_samples}, "
                    f"got {train_metrics['sample_count']}."
                )
            if validation_metrics["sample_count"] != args.num_fixed_eval_samples:
                raise RuntimeError(
                    "Gathered validation FM probe sample count mismatch: "
                    f"expected {args.num_fixed_eval_samples}, "
                    f"got {validation_metrics['sample_count']}."
                )
            record = make_fm_generalization_record(
                iteration=iteration,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                model_weight="raw",
            )
            record.update(
                probe_source_seed=int(args.eval_source_seed),
                probe_time_seed=int(args.fixed_eval_dataset_seed),
            )
            generalization_dir = output_dir / "eval" / "fm_generalization"
            history_path = generalization_dir / "metrics.jsonl"
            append_jsonl(history_path, record)
            save_fm_generalization_curves(
                history_path,
                generalization_dir / "curve_train_validation_loss.png",
            )
            print(format_console_log_record(record), flush=True)
        except Exception as error:
            postprocess_error = error
    _collective_raise_if_failed(
        accelerator,
        postprocess_error,
        "fixed FM generalization postprocessing",
    )


@torch.no_grad()
def _run_fixed_evaluation(
    args: argparse.Namespace,
    accelerator: Accelerator,
    model,
    ema: EMAModel | None,
    loader,
    dataset,
    output_dir: Path,
    iteration: int,
) -> None:
    eval_model = accelerator.unwrap_model(model)
    was_training = model.training
    model.eval()
    ema_stored = False
    try:
        if ema is not None:
            ema.store(eval_model.parameters())
            ema_stored = True
            ema.copy_to(eval_model.parameters())
        sampler = HistogramLinearSampler(
            eval_model,
            accelerator.device,
            source_seed=args.eval_source_seed + accelerator.process_index,
            model_time_scale=args.model_time_scale,
        )
        samples, targets, measurements, diagnostics, elapsed = collect_euler_samples(
            accelerator=accelerator,
            loader=loader,
            sampler=sampler,
            echo_times=dataset.echo_times,
            state_size=args.num_spectrum_bins,
            noise_standard_deviation=args.noise_standard_deviation,
            num_samples_per_condition=args.eval_samples_per_condition,
            num_ode_steps=args.eval_ode_steps,
        )
    finally:
        if ema is not None and ema_stored:
            ema.restore(eval_model.parameters())
        if was_training:
            model.train()
        else:
            model.eval()

    postprocess_error = None
    if accelerator.is_main_process:
        try:
            if samples is None or targets is None or measurements is None:
                raise RuntimeError(
                    "Main process did not receive gathered evaluation tensors."
                )
            if samples.shape[0] != args.num_fixed_eval_samples:
                raise RuntimeError(
                    "Gathered fixed-evaluation condition count mismatch: "
                    f"expected {args.num_fixed_eval_samples}, got {samples.shape[0]}."
                )
            metrics: dict[str, object] = evaluate_histogram_samples(
                samples,
                targets,
                measurements,
                dataset.kernel,
                dataset.log_rate_edges,
                noise_standard_deviation=args.noise_standard_deviation,
                w2_num_quantiles=args.eval_metric_num_quantiles,
                peak_threshold_fraction=args.eval_peak_relative_threshold,
            )
            metrics.update(
                {
                    "iteration": int(iteration),
                    "model_weight": "ema" if ema is not None else "raw",
                    "num_conditions": int(samples.shape[0]),
                    "samples_per_condition": int(samples.shape[1]),
                    "euler_nfe": int(args.eval_ode_steps),
                    "inference_wall_seconds": float(elapsed),
                    "inference_seconds_per_generated_sample": float(elapsed)
                    / float(samples.shape[0] * samples.shape[1]),
                    "pre_projection_negative_fraction": _safe_ratio(
                        diagnostics["pre_projection_negative_value_count"],
                        diagnostics["pre_projection_inspected_value_count"],
                        "pre_projection_negative_fraction",
                    ),
                    "pre_projection_mass_absolute_error_mean": _safe_ratio(
                        diagnostics["pre_projection_mass_error_sum"],
                        diagnostics["pre_projection_inspected_state_count"],
                        "pre_projection_mass_absolute_error_mean",
                    ),
                }
            )
            metrics.update({name: float(value) for name, value in diagnostics.items()})
            fixed_eval_dir = output_dir / "eval" / "fixed_validation"
            metrics_history_path = fixed_eval_dir / "metrics.jsonl"
            append_jsonl(metrics_history_path, metrics)
            write_json(fixed_eval_dir / "latest_metrics.json", metrics)
            qualitative_samples = samples[0, : args.eval_num_visual_samples]
            figure_title = (
                "Histogram-linear | "
                f"iteration {iteration} | Euler NFE {args.eval_ode_steps}"
            )
            save_reconstruction_figure(
                fixed_eval_dir / "latest_reconstruction.png",
                qualitative_samples,
                targets[0],
                measurements[0],
                dataset.rate_centers,
                dataset.echo_times,
                dataset.kernel,
                noise_standard_deviation=args.noise_standard_deviation,
                figure_title=figure_title,
            )
            grid_samples = samples[:16]
            grid_targets = targets[:16]
            save_reconstruction_grid_figure(
                fixed_eval_dir / "latest_reconstruction_grid.png",
                grid_samples,
                grid_targets,
                dataset.rate_centers,
            )
            print(
                format_console_log_record(metrics, indent=2, sort_keys=True),
                flush=True,
            )
            save_training_curves(
                output_dir / "loss_history.jsonl",
                metrics_history_path,
                output_dir / "eval" / "loss" / "curve_loss.png",
            )
        except Exception as error:
            postprocess_error = error
    _collective_raise_if_failed(
        accelerator, postprocess_error, "fixed evaluation postprocessing"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_batches,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)
    if args.source_seed + accelerator.num_processes > 2**63:
        raise ValueError("source_seed plus process rank must lie below 2**63.")
    if args.eval_source_seed + accelerator.num_processes > 2**63:
        raise ValueError("eval_source_seed plus process rank must lie below 2**63.")
    global_eval_batch_size = args.eval_batch_size * accelerator.num_processes
    if args.num_fixed_eval_samples % global_eval_batch_size:
        raise ValueError(
            "num_fixed_eval_samples must be divisible by eval_batch_size * "
            "num_processes so fixed evaluation has no padded conditions."
        )

    output_dir = Path(args.output_dir)
    setup_error = None
    if accelerator.is_main_process:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            setup_error = error
    _collective_raise_if_failed(
        accelerator,
        setup_error,
        "output-directory setup",
    )

    fixed_family_dataset = make_continuous_dataset_family(
        args,
        num_samples=args.num_fixed_eval_samples,
        seed=args.fixed_eval_dataset_seed,
        split="validation",
        refresh_each_epoch=False,
    )
    train_probe_family_dataset = make_continuous_dataset_family(
        args,
        num_samples=args.num_fixed_eval_samples,
        seed=args.train_dataset_seed,
        split="train",
        refresh_each_epoch=False,
    )
    # These fixed IDs are the leading examples of the iteration-addressed
    # training stream, retained only as a stable generalization diagnostic.
    train_probe_target_dataset = train_probe_family_dataset.as_single_target_dataset()
    fixed_dataset = fixed_family_dataset.as_single_target_dataset()
    probe_source_histograms, probe_flow_times = build_fixed_fm_probe_bank(
        num_samples=args.num_fixed_eval_samples,
        num_bins=args.num_spectrum_bins,
        source_seed=args.eval_source_seed,
        time_seed=args.fixed_eval_dataset_seed,
    )
    train_probe_dataset = FixedFMProbeDataset(
        Subset(train_probe_target_dataset, range(args.num_fixed_eval_samples)),
        probe_source_histograms,
        probe_flow_times,
    )
    validation_probe_dataset = FixedFMProbeDataset(
        fixed_dataset,
        probe_source_histograms,
        probe_flow_times,
    )

    fixed_eval_generator = torch.Generator(device="cpu").manual_seed(
        args.fixed_eval_dataset_seed
    )
    fixed_loader = DataLoader(
        fixed_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_dataloader_num_workers,
        drop_last=False,
        generator=fixed_eval_generator,
    )
    train_probe_generator = torch.Generator(device="cpu").manual_seed(
        args.eval_source_seed
    )
    validation_probe_generator = torch.Generator(device="cpu").manual_seed(
        args.eval_source_seed
    )
    train_probe_loader = DataLoader(
        train_probe_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_dataloader_num_workers,
        drop_last=False,
        generator=train_probe_generator,
    )
    validation_probe_loader = DataLoader(
        validation_probe_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_dataloader_num_workers,
        drop_last=False,
        generator=validation_probe_generator,
    )

    model = UNet1DModel(
        **get_unet_config(
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    (
        model,
        optimizer,
        fixed_loader,
        train_probe_loader,
        validation_probe_loader,
    ) = accelerator.prepare(
        model,
        optimizer,
        fixed_loader,
        train_probe_loader,
        validation_probe_loader,
    )
    if len(fixed_loader) == 0:
        raise ValueError("The prepared fixed-evaluation loader has no batches.")
    if len(train_probe_loader) == 0 or len(validation_probe_loader) == 0:
        raise ValueError("The prepared fixed FM probe loaders must have batches.")
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=(args.lr_warmup_iterations * accelerator.num_processes),
        num_training_steps=(args.num_train_iterations * accelerator.num_processes),
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    scheduler = accelerator.prepare(scheduler)

    unwrapped = accelerator.unwrap_model(model)
    ema = None
    if args.use_ema:
        ema = EMAModel(
            unwrapped.parameters(),
            decay=args.ema_decay,
            min_decay=args.ema_min_decay,
            update_after_step=args.ema_update_after_iterations,
            use_ema_warmup=args.ema_use_warmup,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet1DModel,
            model_config=unwrapped.config,
        )
        ema.to(accelerator.device)
        accelerator.register_for_checkpointing(ema)
    trainer = HistogramLinearTrainer(model_time_scale=args.model_time_scale)

    rank_source_seed = args.source_seed + accelerator.process_index
    if rank_source_seed >= 2**63:
        raise ValueError("The rank-specific source seed must lie in [0, 2**63).")
    torch.manual_seed(rank_source_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_source_seed)

    config = dict(vars(args))
    config.update(
        experiment=EXPERIMENT_NAME,
        network_architecture=get_network_architecture_signature(
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
        ),
        world_size=accelerator.num_processes,
        effective_global_batch_size=(
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_batches
        ),
        iteration_definition=(
            "one synchronized optimizer-step boundary (including AMP-skipped attempts)"
        ),
        training_pairs_per_iteration=(
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_batches
        ),
        total_training_pairs=(
            args.num_train_iterations
            * args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_batches
        ),
        **continuous_dataset_config_values(args),
        training_stream_generation={
            "master_seed": int(args.train_dataset_seed),
            "split": "train",
            "infinite_on_the_fly": True,
            "global_pair_ids_are_iteration_addressed": True,
        },
        fixed_train_probe_dataset_generation=json_safe_dataset_metadata(
            train_probe_family_dataset
        ),
        fixed_validation_dataset_generation=json_safe_dataset_metadata(
            fixed_family_dataset
        ),
        state_size=args.num_spectrum_bins,
        model_in_channels=2 * args.num_measurements + 2,
    )

    resume = None
    resume_resolution_error = None
    try:
        resume = _resolve_resume(args)
    except Exception as error:
        resume_resolution_error = error
    _collective_raise_if_failed(
        accelerator,
        resume_resolution_error,
        "checkpoint resolution",
    )

    global_iteration = 0
    resume_validation_error = None
    if resume is not None:
        try:
            state = _read_state(resume)
            _validate_resume(config, state)
            global_iteration = _resume_progress(
                state,
                requested_iterations=args.num_train_iterations,
            )
        except Exception as error:
            resume_validation_error = error
    _collective_raise_if_failed(
        accelerator,
        resume_validation_error,
        "checkpoint metadata validation",
    )

    resume_load_error = None
    if resume is not None:
        try:
            accelerator.load_state(str(resume))
            if ema is not None:
                ema.to(accelerator.device)
        except Exception as error:
            resume_load_error = error
    _collective_raise_if_failed(
        accelerator,
        resume_load_error,
        "checkpoint state loading",
    )
    if resume is not None:
        accelerator.print(
            f"Resumed histogram-linear training from {resume} "
            f"after iteration {global_iteration}."
        )
    if global_iteration == args.num_train_iterations:
        accelerator.print(
            "Histogram-linear training is already complete at iteration "
            f"{global_iteration}."
        )
        accelerator.end_training()
        return

    loss_history_path = output_dir / "loss_history.jsonl"
    fixed_metrics_path = output_dir / "eval" / "fixed_validation" / "metrics.jsonl"
    fm_generalization_history_path = (
        output_dir / "eval" / "fm_generalization" / "metrics.jsonl"
    )
    history_setup_error = None
    if resume is None:
        require_fresh_training_history(output_dir)
    if accelerator.is_main_process:
        try:
            if resume is not None:
                truncate_jsonl_after_iteration(loss_history_path, global_iteration)
                truncate_jsonl_after_iteration(fixed_metrics_path, global_iteration)
                truncate_jsonl_after_iteration(
                    fm_generalization_history_path, global_iteration
                )
                restore_latest_fixed_evaluation_metrics(
                    output_dir / "eval" / "fixed_validation",
                    max_iteration=global_iteration,
                    discard_latest_images=True,
                )
                save_training_curves(
                    loss_history_path,
                    fixed_metrics_path,
                    output_dir / "eval" / "loss" / "curve_loss.png",
                )
                save_fm_generalization_curves(
                    fm_generalization_history_path,
                    output_dir
                    / "eval"
                    / "fm_generalization"
                    / "curve_train_validation_loss.png",
                )
            write_json(output_dir / "training_config.json", config)
        except Exception as error:
            history_setup_error = error
    _collective_raise_if_failed(
        accelerator, history_setup_error, "training-history setup"
    )

    training_stream_family = make_continuous_training_stream(
        args,
        start_iteration=global_iteration,
        local_batch_size=args.train_batch_size,
        gradient_accumulation_batches=args.gradient_accumulation_batches,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=args.train_dataset_seed,
        split="train",
    )
    training_dataset = training_stream_family.as_posterior_weighted_training_stream()
    # This stream is already rank-sharded and already batched. Passing it to
    # accelerator.prepare() would shard it a second time and change pair IDs.
    loader = DataLoader(
        training_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        generator=torch.Generator(device="cpu").manual_seed(args.train_dataset_seed),
    )

    run_started_at = start_distributed_wall_timer(accelerator)
    logging_window_started_at = run_started_at
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    window_normalized_sum = torch.zeros_like(window_loss_sum)
    window_microbatch_count = torch.zeros_like(window_loss_sum)
    window_skipped_optimizer_updates = 0
    microbatches_since_update = 0

    for targets, measurements, sample_weights in loader:
        targets = targets.to(accelerator.device, non_blocking=True)
        measurements = measurements.to(accelerator.device, non_blocking=True)
        sample_weights = sample_weights.to(accelerator.device, non_blocking=True)
        microbatches_since_update += 1
        with accelerator.accumulate(model):
            condition = make_condition(
                measurements,
                training_dataset.echo_times,
                output_size=args.num_spectrum_bins,
                noise_standard_deviation=args.noise_standard_deviation,
            )
            loss, logs = trainer.compute_loss(
                model,
                targets,
                condition,
                sample_weights=sample_weights,
            )
            window_loss_sum += logs["loss"].detach().to(torch.float64)
            window_normalized_sum += logs["normalized_velocity_mse"].detach().to(
                torch.float64
            )
            window_microbatch_count += 1.0
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        if microbatches_since_update != args.gradient_accumulation_batches:
            raise RuntimeError(
                "Training stream and gradient accumulation lost alignment: "
                f"expected {args.gradient_accumulation_batches} microbatches, "
                f"got {microbatches_since_update}."
            )
        microbatches_since_update = 0
        step_was_skipped = accelerator.optimizer_step_was_skipped
        if step_was_skipped:
            window_skipped_optimizer_updates += 1
        if ema is not None and not step_was_skipped:
            ema.step(accelerator.unwrap_model(model).parameters())
        global_iteration += 1
        reached_final_iteration = global_iteration == args.num_train_iterations

        should_log = (
            global_iteration % args.log_every_iterations == 0
            or reached_final_iteration
        )
        if should_log:
            logging_window_wall_seconds = slowest_distributed_elapsed_seconds(
                accelerator, logging_window_started_at
            )
            run_elapsed_wall_seconds = slowest_distributed_elapsed_seconds(
                accelerator, run_started_at
            )
            local_statistics = torch.stack(
                (window_loss_sum, window_normalized_sum, window_microbatch_count)
            ).reshape(1, 3)
            global_statistics = accelerator.gather(local_statistics).sum(dim=0)
            global_microbatch_count = float(global_statistics[2].item())
            if global_microbatch_count <= 0.0:
                raise RuntimeError("Cannot log an empty iteration window.")
            logging_error = None
            if accelerator.is_main_process:
                try:
                    record = {
                        "iteration": int(global_iteration),
                        "loss": float(global_statistics[0].item())
                        / global_microbatch_count,
                        "normalized_velocity_mse": float(global_statistics[1].item())
                        / global_microbatch_count,
                        "microbatch_count": int(global_microbatch_count),
                        "skipped_optimizer_updates": int(
                            window_skipped_optimizer_updates
                        ),
                        "logging_window_wall_seconds": float(
                            logging_window_wall_seconds
                        ),
                        "run_elapsed_wall_seconds": float(run_elapsed_wall_seconds),
                    }
                    append_jsonl(loss_history_path, record)
                    save_training_curves(
                        loss_history_path,
                        fixed_metrics_path,
                        output_dir / "eval" / "loss" / "curve_loss.png",
                    )
                    print(format_console_log_record(record), flush=True)
                except Exception as error:
                    logging_error = error
            _collective_raise_if_failed(
                accelerator, logging_error, "iteration loss-history update"
            )
            window_loss_sum.zero_()
            window_normalized_sum.zero_()
            window_microbatch_count.zero_()
            window_skipped_optimizer_updates = 0
            logging_window_started_at = start_distributed_wall_timer(accelerator)

        should_evaluate = (
            global_iteration % args.eval_every_iterations == 0
            or reached_final_iteration
        )
        if should_evaluate:
            _run_fixed_fm_generalization_evaluation(
                args,
                accelerator,
                model,
                train_probe_loader,
                validation_probe_loader,
                train_probe_target_dataset,
                output_dir,
                global_iteration,
            )
            _run_fixed_evaluation(
                args,
                accelerator,
                model,
                ema,
                fixed_loader,
                fixed_dataset,
                output_dir,
                global_iteration,
            )

        should_checkpoint = (
            global_iteration % args.checkpoint_every_iterations == 0
            or reached_final_iteration
        )
        if should_checkpoint:
            checkpoint = save_latest_accelerator_checkpoint(
                accelerator,
                model,
                ema,
                output_dir,
                iteration=global_iteration,
                training_config=config,
                experiment=EXPERIMENT_NAME,
            )
            accelerator.print(
                "Saved latest checkpoint after iteration "
                f"{global_iteration}: {checkpoint}"
            )
        if reached_final_iteration:
            break

    accelerator.print(
        "Histogram-linear training finished after iteration "
        f"{global_iteration}."
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
