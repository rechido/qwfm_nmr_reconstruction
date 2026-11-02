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

from samplers.conditional_normalizing_flow_sampler import (
    ConditionalNormalizingFlowSampler,
)
from trainers.conditional_normalizing_flow_trainer import (
    ConditionalNormalizingFlowTrainer,
)
from utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    resolve_checkpoint,
    save_latest_accelerator_checkpoint,
    validate_checkpoint_path,
)
from utils.conditional_normalizing_flow_utils import (
    CONDITIONAL_FLOW_INPUT_REPRESENTATION,
    CONDITIONAL_FLOW_MODEL_CLASS,
    CONDITIONAL_FLOW_OUTPUT_PARAMETERIZATION,
    CONDITIONAL_FLOW_TARGET_REPRESENTATION,
    create_conditional_flow_model,
    get_conditional_flow_architecture_signature,
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


EXPERIMENT_NAME = "conditional_normalizing_flow"
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
        description=(
            "Train a conditional RealNVP baseline in an additive-log-ratio "
            "chart of the NMR spectrum simplex."
        )
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
    parser.add_argument("--resume_from_checkpoint", type=str, required=True)
    parser.add_argument("--num_coupling_layers", type=int, default=8)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--condition_context_dim", type=int, default=256)
    parser.add_argument("--num_condition_blocks", type=int, default=3)
    parser.add_argument("--coupling_scale_limit", type=float, default=1.5)
    parser.add_argument("--simplex_interior_mass", type=float, default=1.0e-4)
    parser.add_argument("--alr_dequantization_std", type=float, default=0.02)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    positive_integers = (
        "num_measurements",
        "num_spectrum_bins",
        "train_batch_size",
        "num_train_iterations",
        "gradient_accumulation_batches",
        "log_every_iterations",
        "eval_every_iterations",
        "num_fixed_eval_samples",
        "eval_batch_size",
        "eval_samples_per_condition",
        "eval_num_visual_samples",
        "eval_metric_num_quantiles",
        "checkpoint_every_iterations",
        "lr_num_cycles",
        "num_coupling_layers",
        "flow_hidden_dim",
        "condition_context_dim",
        "num_condition_blocks",
    )
    for name in positive_integers:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in (
        "dataloader_num_workers",
        "eval_dataloader_num_workers",
        "lr_warmup_iterations",
        "ema_update_after_iterations",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    for name in (
        "train_dataset_seed",
        "fixed_eval_dataset_seed",
        "eval_source_seed",
        "seed",
    ):
        if not 0 <= int(getattr(args, name)) < 2**63:
            raise ValueError(f"{name} must lie in [0, 2**63).")
    if args.num_measurements < 2:
        raise ValueError("num_measurements must be at least two.")
    if args.num_spectrum_bins < 2:
        raise ValueError("num_spectrum_bins must be at least two.")
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
    if not 0.0 <= args.eval_peak_relative_threshold <= 1.0:
        raise ValueError("eval_peak_relative_threshold must lie in [0, 1].")
    if args.learning_rate <= 0.0 or args.adam_epsilon <= 0.0:
        raise ValueError("learning_rate and adam_epsilon must be positive.")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must lie in [0, 1).")
    if args.adam_weight_decay < 0.0 or args.max_grad_norm <= 0.0:
        raise ValueError("Weight decay must be nonnegative and grad norm positive.")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must lie in (0, 1).")
    if not 0.0 <= args.ema_min_decay <= args.ema_decay:
        raise ValueError("ema_min_decay must lie in [0, ema_decay].")
    if args.ema_inv_gamma <= 0.0 or args.ema_power <= 0.0:
        raise ValueError("EMA warmup values must be positive.")
    if not 0.0 < args.simplex_interior_mass < 1.0:
        raise ValueError("simplex_interior_mass must lie in (0, 1).")
    if (
        not math.isfinite(args.alr_dequantization_std)
        or args.alr_dequantization_std < 0
    ):
        raise ValueError("alr_dequantization_std must be finite and nonnegative.")
    if args.coupling_scale_limit <= 0.0:
        raise ValueError("coupling_scale_limit must be positive.")
    if not args.output_dir.strip() or not args.resume_from_checkpoint.strip():
        raise ValueError("Output and resume arguments must not be empty.")


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


def _read_state(checkpoint: Path) -> dict[str, Any]:
    with (checkpoint / "training_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint training_state.json must contain an object.")
    return state


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
    iteration = state.get("global_iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Checkpoint global_iteration must be nonnegative.")
    if iteration > requested_iterations:
        raise ValueError("Checkpoint is beyond the requested training horizon.")
    return int(iteration)


def _architecture_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_spectrum_bins": args.num_spectrum_bins,
        "num_measurements": args.num_measurements,
        "num_coupling_layers": args.num_coupling_layers,
        "hidden_dim": args.flow_hidden_dim,
        "context_dim": args.condition_context_dim,
        "num_condition_blocks": args.num_condition_blocks,
        "scale_limit": args.coupling_scale_limit,
    }


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
    evaluation_model = accelerator.unwrap_model(model)
    was_training = model.training
    ema_stored = False
    sample_batches: list[torch.Tensor] = []
    reference_batches: list[torch.Tensor] = []
    weight_batches: list[torch.Tensor] = []
    measurement_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    try:
        if ema is not None:
            ema.store(evaluation_model.parameters())
            ema.copy_to(evaluation_model.parameters())
            ema_stored = True
        model.eval()
        sampler = ConditionalNormalizingFlowSampler(
            evaluation_model,
            accelerator.device,
            source_seed=args.eval_source_seed + accelerator.process_index,
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
                samples, batch_diagnostics = sampler.sample(
                    condition,
                    args.eval_samples_per_condition,
                )
            gathered = accelerator.gather_for_metrics(
                (
                    samples,
                    batch["reference_histograms"],
                    batch["posterior_weights"],
                    measurements,
                    batch["target_histogram"],
                )
            )
            if accelerator.is_main_process:
                sample, reference, weight, measurement, target = gathered
                sample_batches.append(sample.detach().cpu())
                reference_batches.append(reference.detach().cpu())
                weight_batches.append(weight.detach().cpu())
                measurement_batches.append(measurement.detach().cpu())
                target_batches.append(target.detach().cpu())
            for name, value in batch_diagnostics.items():
                reduced = accelerator.reduce(value, reduction="sum")
                if accelerator.is_main_process:
                    diagnostics[name] = diagnostics.get(name, 0.0) + float(
                        reduced.item()
                    )
        elapsed = slowest_distributed_elapsed_seconds(accelerator, started)
    finally:
        if ema_stored:
            ema.restore(evaluation_model.parameters())
        model.train(was_training)

    error: Exception | None = None
    if accelerator.is_main_process:
        try:
            samples = torch.cat(sample_batches)
            references = torch.cat(reference_batches)
            weights = torch.cat(weight_batches)
            measurements = torch.cat(measurement_batches)
            targets = torch.cat(target_batches)
            del references, weights
            sampled_target = evaluate_histogram_samples(
                samples,
                targets,
                measurements,
                dataset.kernel,
                dataset.log_rate_edges,
                noise_standard_deviation=args.noise_standard_deviation,
                w2_num_quantiles=args.eval_metric_num_quantiles,
                peak_threshold_fraction=args.eval_peak_relative_threshold,
            )
            generated_count = float(diagnostics["generated_state_count"])
            record = {
                "iteration": int(iteration),
                "model_weight": "ema" if ema is not None else "raw",
                **sampled_target,
                "fixed_validation_metric_protocol": FIXED_VALIDATION_METRIC_PROTOCOL,
                "num_conditions": int(samples.shape[0]),
                "samples_per_condition": int(samples.shape[1]),
                "actual_model_nfe_per_generated_sample": 1,
                "inverse_coupling_layers_per_generated_sample": int(
                    evaluation_model.config.num_coupling_layers
                ),
                "inference_wall_seconds": float(elapsed),
                "inference_seconds_per_generated_sample": float(elapsed)
                / generated_count,
                "sample_negative_fraction": diagnostics["negative_value_count"]
                / diagnostics["generated_value_count"],
                "sample_nonfinite_fraction": diagnostics["nonfinite_value_count"]
                / diagnostics["generated_value_count"],
                "sample_mass_absolute_error_mean": diagnostics[
                    "mass_absolute_error_sum"
                ]
                / generated_count,
                "sample_alr_absolute_value_mean": diagnostics["alr_absolute_value_sum"]
                / diagnostics["alr_value_count"],
            }
            directory = output_dir / "eval" / "fixed_validation"
            history_path = directory / "metrics.jsonl"
            append_jsonl(history_path, record)
            write_json(directory / "latest_metrics.json", record)
            save_reconstruction_figure(
                directory / "latest_reconstruction.png",
                samples[0, : args.eval_num_visual_samples],
                targets[0],
                measurements[0],
                dataset.rate_centers,
                dataset.echo_times,
                dataset.kernel,
                noise_standard_deviation=args.noise_standard_deviation,
                figure_title=(
                    f"Conditional normalizing flow | iteration {iteration} | "
                    f"K={args.eval_samples_per_condition}"
                ),
            )
            save_reconstruction_grid_figure(
                directory / "latest_reconstruction_grid.png",
                samples[:16],
                targets[:16],
                dataset.rate_centers,
            )
            save_training_curves(
                output_dir / "loss_history.jsonl",
                history_path,
                output_dir / "eval" / "loss" / "curve_loss.png",
            )
            print(
                format_console_log_record(record, indent=2, sort_keys=True), flush=True
            )
        except Exception as caught:
            error = caught
    _collective_raise_if_failed(accelerator, error, "CNF fixed evaluation")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_batches,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)
    if args.eval_source_seed + accelerator.num_processes > 2**63:
        raise ValueError("eval_source_seed plus process rank must lie below 2**63.")
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
        except Exception as caught:
            setup_error = caught
    _collective_raise_if_failed(accelerator, setup_error, "CNF output setup")

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

    architecture = _architecture_from_args(args)
    model = create_conditional_flow_model(**architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    model, optimizer, fixed_loader = accelerator.prepare(model, optimizer, fixed_loader)
    if not len(fixed_loader):
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
            model_cls=CONDITIONAL_FLOW_MODEL_CLASS,
            model_config=unwrapped.config,
        )
        ema.to(accelerator.device)
        accelerator.register_for_checkpointing(ema)
    trainer = ConditionalNormalizingFlowTrainer(
        interior_mass=args.simplex_interior_mass,
        alr_dequantization_std=args.alr_dequantization_std,
    )

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
        network_architecture=get_conditional_flow_architecture_signature(
            **architecture
        ),
        conditional_input_representation=CONDITIONAL_FLOW_INPUT_REPRESENTATION,
        conditional_output_parameterization=CONDITIONAL_FLOW_OUTPUT_PARAMETERIZATION,
        conditional_target_representation=CONDITIONAL_FLOW_TARGET_REPRESENTATION,
        alr_reference_component="last_spectrum_bin",
        likelihood_measure="Lebesgue_measure_in_fixed_ALR_chart",
        model_parameter_count=sum(
            parameter.numel() for parameter in unwrapped.parameters()
        ),
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
    resume: Path | None = None
    try:
        resume = _resolve_resume(args)
    except Exception as caught:
        resume_error = caught
    _collective_raise_if_failed(accelerator, resume_error, "CNF checkpoint resolution")
    global_iteration = 0
    metadata_error: Exception | None = None
    if resume is not None:
        try:
            state = _read_state(resume)
            if state.get("experiment") != EXPERIMENT_NAME:
                raise ValueError("Checkpoint experiment does not match CNF.")
            _validate_resume_config(config, state)
            global_iteration = _resume_progress(
                state,
                requested_iterations=args.num_train_iterations,
            )
        except Exception as caught:
            metadata_error = caught
    _collective_raise_if_failed(accelerator, metadata_error, "CNF resume metadata")
    load_error: Exception | None = None
    if resume is not None:
        try:
            accelerator.load_state(str(resume))
            if ema is not None:
                ema.to(accelerator.device)
        except Exception as caught:
            load_error = caught
    _collective_raise_if_failed(accelerator, load_error, "CNF checkpoint loading")

    history_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            if resume is None:
                require_fresh_training_history(output_dir)
            else:
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
                    discard_latest_images=True,
                )
            write_json(output_dir / "training_config.json", config)
            save_training_curves(
                output_dir / "loss_history.jsonl",
                output_dir / "eval" / "fixed_validation" / "metrics.jsonl",
                output_dir / "eval" / "loss" / "curve_loss.png",
            )
        except Exception as caught:
            history_error = caught
    _collective_raise_if_failed(accelerator, history_error, "CNF history setup")

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
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    nll_sum = torch.zeros_like(loss_sum)
    batch_count = torch.zeros_like(loss_sum)
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
                loss, logs = trainer.compute_loss(
                    model,
                    targets,
                    condition,
                )
                loss_sum += logs["loss"].detach().to(torch.float64)
                nll_sum += logs["conditional_nll"].detach().to(torch.float64)
                batch_count += 1.0
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
            gathered = accelerator.gather(
                torch.stack((loss_sum, nll_sum, batch_count))[None, :]
            ).sum(dim=0)
            logging_interval_wall_seconds = slowest_distributed_elapsed_seconds(
                accelerator, logging_interval_started
            )
            run_elapsed = slowest_distributed_elapsed_seconds(accelerator, run_started)
            record_error: Exception | None = None
            if accelerator.is_main_process:
                try:
                    denominator = float(gathered[2].item())
                    record = {
                        "iteration": iteration,
                        "loss": float(gathered[0].item()) / denominator,
                        "conditional_nll": float(gathered[1].item()) / denominator,
                        "microbatch_count": int(gathered[2].item()),
                        "skipped_optimizer_updates": skipped_updates,
                        "logging_interval_wall_seconds": (
                            logging_interval_wall_seconds
                        ),
                        "run_elapsed_wall_seconds": run_elapsed,
                    }
                    append_jsonl(output_dir / "loss_history.jsonl", record)
                    save_training_curves(
                        output_dir / "loss_history.jsonl",
                        output_dir / "eval" / "fixed_validation" / "metrics.jsonl",
                        output_dir / "eval" / "loss" / "curve_loss.png",
                    )
                    print(format_console_log_record(record), flush=True)
                except Exception as caught:
                    record_error = caught
            _collective_raise_if_failed(
                accelerator, record_error, "CNF iteration logging"
            )
            loss_sum.zero_()
            nll_sum.zero_()
            batch_count.zero_()
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
                f"Saved CNF checkpoint after iteration {iteration}: {checkpoint}"
            )

    accelerator.print(f"CNF training finished after iteration {global_iteration}.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
