# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from samplers.conditional_vae_sampler import ConditionalVAESampler
from trainers.conditional_vae_trainer import ConditionalVAETrainer
from utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    resolve_checkpoint,
    save_latest_accelerator_checkpoint,
    validate_checkpoint_path,
)
from utils.conditional_vae_utils import (
    CONDITIONAL_VAE_INPUT_REPRESENTATION,
    CONDITIONAL_VAE_OUTPUT_PARAMETERIZATION,
    CONDITIONAL_VAE_PRIOR,
    CONDITIONAL_VAE_TIMESTEP,
    ConditionalVAEModel,
    create_conditional_vae_model,
    get_conditional_vae_architecture_signature,
)
from utils.continuous_experiment_utils import (
    add_continuous_dataset_arguments,
    continuous_dataset_config_values,
    json_safe_dataset_metadata,
    make_continuous_dataset_family,
    make_continuous_training_stream,
    validate_continuous_dataset_arguments,
)
from utils.nmr_utils import (
    evaluate_histogram_samples,
    make_condition,
    save_reconstruction_figure,
    save_reconstruction_grid_figure,
    write_json,
)
from utils.training_utils import (
    append_jsonl,
    format_console_log_record,
    nested_config_values_match,
    require_fresh_training_history,
    restore_latest_fixed_evaluation_metrics,
    save_training_curves,
    slowest_distributed_elapsed_seconds,
    start_distributed_wall_timer,
    truncate_jsonl_after_iteration,
)


EXPERIMENT_NAME = "conditional_vae"
FIXED_VALIDATION_METRIC_PROTOCOL = "continuous_fixed_point_target_v1"


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, 1/0, or yes/no.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conditional VAE on posterior-weighted NMR endpoints."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset_seed", type=int, required=True)
    add_continuous_dataset_arguments(parser)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--encoder_hidden_dim", type=int, required=True)
    parser.add_argument("--encoder_num_layers", type=int, required=True)
    parser.add_argument("--kl_weight", type=float, required=True)
    parser.add_argument("--kl_warmup_iterations", type=int, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--num_train_iterations", type=int, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, required=True)
    parser.add_argument("--gradient_accumulation_batches", type=int, required=True)
    parser.add_argument("--log_every_iterations", type=int, required=True)
    parser.add_argument("--eval_every_iterations", type=int, required=True)
    parser.add_argument("--num_fixed_eval_samples", type=int, required=True)
    parser.add_argument("--fixed_eval_dataset_seed", type=int, required=True)
    parser.add_argument("--fixed_eval_sample_seed", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, required=True)
    parser.add_argument("--eval_dataloader_num_workers", type=int, required=True)
    parser.add_argument("--eval_samples_per_condition", type=int, required=True)
    parser.add_argument("--eval_num_visual_samples", type=int, required=True)
    parser.add_argument("--eval_max_decode_batch_size", type=int, required=True)
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
    parser.add_argument("--resume_from_checkpoint", type=str, required=True)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_measurements",
        "num_spectrum_bins",
        "latent_dim",
        "encoder_hidden_dim",
        "encoder_num_layers",
        "train_batch_size",
        "num_train_iterations",
        "gradient_accumulation_batches",
        "log_every_iterations",
        "eval_every_iterations",
        "num_fixed_eval_samples",
        "eval_batch_size",
        "eval_samples_per_condition",
        "eval_num_visual_samples",
        "eval_max_decode_batch_size",
        "eval_metric_num_quantiles",
        "lr_num_cycles",
        "checkpoint_every_iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in (
        "dataloader_num_workers",
        "eval_dataloader_num_workers",
        "kl_warmup_iterations",
        "lr_warmup_iterations",
        "ema_update_after_iterations",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    for name in (
        "train_dataset_seed",
        "fixed_eval_dataset_seed",
        "fixed_eval_sample_seed",
        "seed",
    ):
        if not 0 <= int(getattr(args, name)) < 2**63:
            raise ValueError(f"{name} must lie in [0, 2**63).")
    if args.num_measurements < 2:
        raise ValueError("num_measurements must be at least two.")
    if args.num_spectrum_bins < 8:
        raise ValueError("num_spectrum_bins must be at least eight.")
    if not 0.0 < args.rate_minimum < args.rate_maximum:
        raise ValueError("Require 0 < rate_minimum < rate_maximum.")
    if not 0.0 < args.echo_time_minimum < args.echo_time_maximum:
        raise ValueError("Require 0 < echo_time_minimum < echo_time_maximum.")
    if args.noise_standard_deviation <= 0.0:
        raise ValueError("noise_standard_deviation must be positive.")
    validate_continuous_dataset_arguments(args)
    if args.num_fixed_eval_samples < 16:
        raise ValueError("num_fixed_eval_samples must be at least 16.")
    if args.eval_num_visual_samples > args.eval_samples_per_condition:
        raise ValueError(
            "eval_num_visual_samples cannot exceed eval_samples_per_condition."
        )
    for name in (
        "kl_weight",
        "learning_rate",
        "adam_epsilon",
        "lr_power",
        "max_grad_norm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not 0.0 <= args.eval_peak_relative_threshold <= 1.0:
        raise ValueError("eval_peak_relative_threshold must lie in [0,1].")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must lie in [0,1).")
    if args.adam_weight_decay < 0.0:
        raise ValueError("adam_weight_decay must be nonnegative.")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must lie in (0,1).")
    if not 0.0 <= args.ema_min_decay <= args.ema_decay:
        raise ValueError("ema_min_decay must lie in [0,ema_decay].")
    if args.ema_inv_gamma <= 0.0 or args.ema_power <= 0.0:
        raise ValueError("EMA inv_gamma and power must be positive.")
    if not args.output_dir.strip():
        raise ValueError("output_dir must not be empty.")


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
    count = int(accelerator.reduce(failure, reduction="sum").item())
    if count:
        if local_error is None:
            raise RuntimeError(f"{phase} failed on {count} other process(es).")
        raise RuntimeError(f"{phase} failed: {local_error}") from local_error


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
                    "Checkpoint artifacts exist, but none is a complete state."
                ) from error
            return None
    return validate_checkpoint_path(
        requested,
        require_ema=args.use_ema,
        allow_legacy_numbered=False,
    )


def _read_state(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "training_state.json"
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError("training_state.json must contain an object.")
    return state


def _validate_resume_config(current: dict[str, Any], state: dict[str, Any]) -> None:
    saved = state.get("training_config")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint lacks a complete training_config.")
    mutable = {
        "output_dir",
        "num_train_iterations",
        "dataloader_num_workers",
        "log_every_iterations",
        "eval_every_iterations",
        "eval_dataloader_num_workers",
        "checkpoint_every_iterations",
        "resume_from_checkpoint",
        "total_training_pairs",
    }
    mismatches = [
        f"{name}: saved={saved.get(name)!r}, current={value!r}"
        for name, value in current.items()
        if name not in mutable
        and not nested_config_values_match(value, saved.get(name))
    ]
    if mismatches:
        raise ValueError("Immutable checkpoint mismatch:\n" + "\n".join(mismatches))


def _resume_progress(
    state: dict[str, Any],
    *,
    requested_iterations: int,
) -> int:
    if state.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Only iteration-native checkpoint schema "
            f"v{CHECKPOINT_SCHEMA_VERSION} can resume this training stream."
        )
    if state.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint experiment is not conditional_vae.")
    iteration = state.get("global_iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Checkpoint global_iteration must be nonnegative.")
    if iteration > requested_iterations:
        raise ValueError("Checkpoint is beyond the requested training horizon.")
    return int(iteration)


def _kl_multiplier(global_iteration: int, warmup_updates: int) -> float:
    if warmup_updates <= 0:
        return 1.0
    return min(1.0, float(global_iteration + 1) / float(warmup_updates))


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
    unwrapped = accelerator.unwrap_model(model)
    was_training = model.training
    ema_stored = False
    prediction_batches: list[torch.Tensor] = []
    reference_batches: list[torch.Tensor] = []
    weight_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    measurement_batches: list[torch.Tensor] = []
    try:
        if ema is not None:
            ema.store(unwrapped.parameters())
            ema.copy_to(unwrapped.parameters())
            ema_stored = True
        model.eval()
        sampler = ConditionalVAESampler(
            unwrapped,
            max_decode_batch_size=args.eval_max_decode_batch_size,
        )
        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.fixed_eval_sample_seed + accelerator.process_index
        )
        started = start_distributed_wall_timer(accelerator)
        for batch in loader:
            measurements = batch["measurement"]
            condition = make_condition(
                measurements,
                dataset.echo_times,
                output_size=args.num_spectrum_bins,
                noise_standard_deviation=args.noise_standard_deviation,
            )
            with accelerator.autocast():
                predictions, _ = sampler.sample(
                    condition,
                    args.eval_samples_per_condition,
                    generator=generator,
                )
            gathered = accelerator.gather_for_metrics(
                (
                    predictions,
                    batch["reference_histograms"],
                    batch["posterior_weights"],
                    batch["target_histogram"],
                    measurements,
                )
            )
            if accelerator.is_main_process:
                prediction, references, weights, targets, measurement = gathered
                prediction_batches.append(prediction.detach().cpu())
                reference_batches.append(references.detach().cpu())
                weight_batches.append(weights.detach().cpu())
                target_batches.append(targets.detach().cpu())
                measurement_batches.append(measurement.detach().cpu())
        elapsed = slowest_distributed_elapsed_seconds(accelerator, started)
    finally:
        if ema_stored:
            ema.restore(unwrapped.parameters())
        model.train(was_training)

    postprocess_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            predictions = torch.cat(prediction_batches)
            references = torch.cat(reference_batches)
            weights = torch.cat(weight_batches)
            targets = torch.cat(target_batches)
            measurements = torch.cat(measurement_batches)
            del references, weights
            sampled_target = evaluate_histogram_samples(
                predictions,
                targets,
                measurements,
                dataset.kernel,
                dataset.log_rate_edges,
                noise_standard_deviation=args.noise_standard_deviation,
                w2_num_quantiles=args.eval_metric_num_quantiles,
                peak_threshold_fraction=args.eval_peak_relative_threshold,
            )
            record = {
                "iteration": int(iteration),
                "model_weight": "ema" if ema is not None else "raw",
                **sampled_target,
                "fixed_validation_metric_protocol": FIXED_VALIDATION_METRIC_PROTOCOL,
                "num_conditions": int(predictions.shape[0]),
                "samples_per_condition": int(predictions.shape[1]),
                "unique_model_predictions_per_condition": int(predictions.shape[1]),
                "actual_model_nfe_per_condition": int(predictions.shape[1]),
                "inference_wall_seconds": float(elapsed),
                "inference_seconds_per_generated_sample": float(elapsed)
                / float(predictions.shape[0] * predictions.shape[1]),
            }
            evaluation_dir = output_dir / "eval" / "fixed_validation"
            metrics_path = evaluation_dir / "metrics.jsonl"
            append_jsonl(metrics_path, record)
            write_json(evaluation_dir / "latest_metrics.json", record)
            save_reconstruction_figure(
                evaluation_dir / "latest_reconstruction.png",
                predictions[0, : args.eval_num_visual_samples],
                targets[0],
                measurements[0],
                dataset.rate_centers,
                dataset.echo_times,
                dataset.kernel,
                noise_standard_deviation=args.noise_standard_deviation,
                figure_title=f"Conditional VAE | iteration {iteration}",
            )
            save_reconstruction_grid_figure(
                evaluation_dir / "latest_reconstruction_grid.png",
                predictions[:16],
                targets[:16],
                dataset.rate_centers,
            )
            save_training_curves(
                output_dir / "loss_history.jsonl",
                metrics_path,
                output_dir / "eval" / "loss" / "curve_loss.png",
            )
            print(
                format_console_log_record(record, indent=2, sort_keys=True), flush=True
            )
        except Exception as error:
            postprocess_error = error
    _collective_raise_if_failed(
        accelerator,
        postprocess_error,
        "conditional-VAE fixed-evaluation postprocessing",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_batches,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)
    if args.fixed_eval_sample_seed + accelerator.num_processes > 2**63:
        raise ValueError(
            "fixed_eval_sample_seed plus process rank must lie below 2**63."
        )
    global_eval_batch = args.eval_batch_size * accelerator.num_processes
    if args.num_fixed_eval_samples % global_eval_batch:
        raise ValueError(
            "num_fixed_eval_samples must be divisible by eval_batch_size * world size."
        )

    output_dir = Path(args.output_dir)
    setup_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            setup_error = error
    _collective_raise_if_failed(accelerator, setup_error, "CVAE output setup")

    fixed_family = make_continuous_dataset_family(
        args,
        num_samples=args.num_fixed_eval_samples,
        seed=args.fixed_eval_dataset_seed,
        split="validation",
        refresh_each_epoch=False,
    )
    fixed_dataset = fixed_family.as_family_evaluation_dataset()
    fixed_loader = DataLoader(
        fixed_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_dataloader_num_workers,
    )

    model = create_conditional_vae_model(
        sample_size=args.num_spectrum_bins,
        num_measurements=args.num_measurements,
        latent_dim=args.latent_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_num_layers=args.encoder_num_layers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    model, optimizer, fixed_loader = accelerator.prepare(model, optimizer, fixed_loader)
    if len(fixed_loader) == 0:
        raise ValueError("Prepared fixed-evaluation loader needs batches.")
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_iterations * accelerator.num_processes,
        num_training_steps=args.num_train_iterations * accelerator.num_processes,
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
            model_cls=ConditionalVAEModel,
            model_config=unwrapped.config,
        )
        ema.to(accelerator.device)
        accelerator.register_for_checkpointing(ema)
    trainer = ConditionalVAETrainer(kl_weight=args.kl_weight)

    config = dict(vars(args))
    config.update(
        experiment=EXPERIMENT_NAME,
        **continuous_dataset_config_values(args),
        training_stream_generation={
            "master_seed": int(args.train_dataset_seed),
            "split": "train",
            "infinite_on_the_fly": True,
            "global_pair_ids_are_iteration_addressed": True,
        },
        network_architecture=get_conditional_vae_architecture_signature(
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
            latent_dim=args.latent_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_num_layers=args.encoder_num_layers,
        ),
        conditional_vae_input_representation=CONDITIONAL_VAE_INPUT_REPRESENTATION,
        conditional_vae_output_parameterization=(
            CONDITIONAL_VAE_OUTPUT_PARAMETERIZATION
        ),
        conditional_vae_prior=CONDITIONAL_VAE_PRIOR,
        conditional_vae_timestep=CONDITIONAL_VAE_TIMESTEP,
        reconstruction_loss="mean_squared_histogram_error_v1",
        kl_reduction="mean_over_latent_coordinates_v1",
        model_parameter_count=sum(p.numel() for p in unwrapped.parameters()),
        world_size=accelerator.num_processes,
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
        effective_global_batch_size=(
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_batches
        ),
        fixed_validation_dataset_generation=json_safe_dataset_metadata(fixed_family),
    )

    resume_error: Exception | None = None
    resume = None
    try:
        resume = _resolve_resume(args)
    except Exception as error:
        resume_error = error
    _collective_raise_if_failed(accelerator, resume_error, "CVAE checkpoint resolution")
    global_iteration = 0
    validation_error: Exception | None = None
    if resume is not None:
        try:
            state = _read_state(resume)
            _validate_resume_config(config, state)
            global_iteration = _resume_progress(
                state,
                requested_iterations=args.num_train_iterations,
            )
        except Exception as error:
            validation_error = error
    _collective_raise_if_failed(
        accelerator, validation_error, "CVAE checkpoint metadata validation"
    )
    load_error: Exception | None = None
    if resume is not None:
        try:
            accelerator.load_state(str(resume))
            if ema is not None:
                ema.to(accelerator.device)
        except Exception as error:
            load_error = error
    _collective_raise_if_failed(accelerator, load_error, "CVAE state loading")

    history_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            if resume is None:
                require_fresh_training_history(output_dir)
            write_json(output_dir / "training_config.json", config)
            truncate_jsonl_after_iteration(
                output_dir / "loss_history.jsonl",
                max_iteration=global_iteration,
            )
            truncate_jsonl_after_iteration(
                output_dir / "eval" / "fixed_validation" / "metrics.jsonl",
                max_iteration=global_iteration,
            )
            restore_latest_fixed_evaluation_metrics(
                output_dir / "eval" / "fixed_validation",
                max_iteration=global_iteration,
                discard_latest_images=resume is not None,
            )
            save_training_curves(
                output_dir / "loss_history.jsonl",
                output_dir / "eval" / "fixed_validation" / "metrics.jsonl",
                output_dir / "eval" / "loss" / "curve_loss.png",
            )
        except Exception as error:
            history_error = error
    _collective_raise_if_failed(accelerator, history_error, "CVAE history setup")

    train_family = make_continuous_training_stream(
        args,
        start_iteration=global_iteration,
        local_batch_size=args.train_batch_size,
        gradient_accumulation_batches=args.gradient_accumulation_batches,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=args.train_dataset_seed,
    )
    train_dataset = train_family.as_single_target_stream()
    train_loader_generator = torch.Generator().manual_seed(args.train_dataset_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        generator=train_loader_generator,
    )
    train_iterator = iter(train_loader)

    run_started = start_distributed_wall_timer(accelerator)
    logging_interval_started = start_distributed_wall_timer(accelerator)
    sums = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
    skipped_updates = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for iteration in range(global_iteration + 1, args.num_train_iterations + 1):
        for accumulation_index in range(args.gradient_accumulation_batches):
            targets, measurements = next(train_iterator)
            targets = targets.to(accelerator.device, non_blocking=True)
            measurements = measurements.to(accelerator.device, non_blocking=True)
            with accelerator.accumulate(model):
                condition = make_condition(
                    measurements,
                    train_family.echo_times,
                    output_size=args.num_spectrum_bins,
                    noise_standard_deviation=args.noise_standard_deviation,
                )
                multiplier = _kl_multiplier(iteration - 1, args.kl_warmup_iterations)
                loss, logs = trainer.compute_loss(
                    model,
                    targets,
                    condition,
                    kl_multiplier=multiplier,
                )
                sums += torch.stack(
                    (
                        logs["loss"].detach(),
                        logs["weighted_reconstruction_mse"].detach(),
                        logs["weighted_kl_per_latent"].detach(),
                        torch.ones((), device=accelerator.device),
                    )
                ).to(torch.float64)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                if accumulation_index + 1 != args.gradient_accumulation_batches:
                    raise RuntimeError(
                        "Gradient synchronization occurred before the final "
                        "microbatch of an optimizer iteration."
                    )
                skipped = accelerator.optimizer_step_was_skipped
                skipped_updates += int(skipped)
                if ema is not None and not skipped:
                    ema.step(accelerator.unwrap_model(model).parameters())
        if not accelerator.sync_gradients:
            raise RuntimeError(
                "An optimizer iteration ended before gradient synchronization."
            )
        global_iteration = iteration
        is_final_iteration = iteration == args.num_train_iterations
        should_log = iteration % args.log_every_iterations == 0 or is_final_iteration
        if should_log:
            gathered = accelerator.gather(sums[None, :]).sum(dim=0)
            logging_interval_wall_seconds = slowest_distributed_elapsed_seconds(
                accelerator, logging_interval_started
            )
            run_wall = slowest_distributed_elapsed_seconds(accelerator, run_started)
            logging_error: Exception | None = None
            if accelerator.is_main_process:
                try:
                    denominator = float(gathered[3].item())
                    record = {
                        "iteration": iteration,
                        "loss": float(gathered[0].item()) / denominator,
                        "weighted_reconstruction_mse": float(gathered[1].item())
                        / denominator,
                        "weighted_kl_per_latent": float(gathered[2].item())
                        / denominator,
                        "microbatch_count": int(gathered[3].item()),
                        "effective_kl_weight": args.kl_weight
                        * _kl_multiplier(iteration - 1, args.kl_warmup_iterations),
                        "skipped_optimizer_updates": skipped_updates,
                        "logging_interval_wall_seconds": (
                            logging_interval_wall_seconds
                        ),
                        "run_elapsed_wall_seconds": run_wall,
                    }
                    append_jsonl(output_dir / "loss_history.jsonl", record)
                    save_training_curves(
                        output_dir / "loss_history.jsonl",
                        output_dir / "eval" / "fixed_validation" / "metrics.jsonl",
                        output_dir / "eval" / "loss" / "curve_loss.png",
                    )
                    print(format_console_log_record(record), flush=True)
                except Exception as error:
                    logging_error = error
            _collective_raise_if_failed(
                accelerator, logging_error, "CVAE iteration logging"
            )
            sums.zero_()
            skipped_updates = 0
            logging_interval_started = start_distributed_wall_timer(accelerator)

        if iteration % args.eval_every_iterations == 0 or is_final_iteration:
            _run_fixed_evaluation(
                args,
                accelerator,
                model,
                ema,
                fixed_loader,
                fixed_dataset,
                output_dir,
                iteration,
            )
        if iteration % args.checkpoint_every_iterations == 0 or is_final_iteration:
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
                f"Saved conditional-VAE checkpoint after iteration {iteration}: "
                f"{checkpoint}"
            )
    accelerator.print(
        f"Conditional-VAE training finished after iteration {global_iteration}."
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
