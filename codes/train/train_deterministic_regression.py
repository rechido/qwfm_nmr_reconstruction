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
from diffusers import UNet1DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from samplers.deterministic_regression_sampler import DeterministicRegressionSampler
from trainers.deterministic_regression_trainer import DeterministicRegressionTrainer
from utils.checkpoint_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    resolve_checkpoint,
    save_latest_accelerator_checkpoint,
    validate_checkpoint_path,
)
from utils.deterministic_regression_utils import (
    DETERMINISTIC_REGRESSION_INPUT_REPRESENTATION,
    DETERMINISTIC_REGRESSION_OUTPUT_PARAMETERIZATION,
    DETERMINISTIC_REGRESSION_TIMESTEP,
    REGRESSION_LOSS_KINDS,
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
from utils.unet_config import (
    get_unet_config,
    get_network_architecture_signature,
)
from utils.training_utils import (
    append_jsonl,
    format_console_log_record,
    require_fresh_training_history,
    restore_latest_fixed_evaluation_metrics,
    save_training_curves,
    slowest_distributed_elapsed_seconds,
    start_distributed_wall_timer,
    truncate_jsonl_after_iteration,
)


EXPERIMENT_NAME = "deterministic_regression"
MSE_W1_EXPERIMENT_NAME = "deterministic_regression_mse_w1"
FIXED_VALIDATION_METRIC_PROTOCOL = "continuous_fixed_point_target_v1"


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


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, 1/0, or yes/no.")


def _experiment_name(args: argparse.Namespace) -> str:
    return (
        MSE_W1_EXPERIMENT_NAME if args.regression_loss == "mse_w1" else EXPERIMENT_NAME
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the matched-backbone deterministic NMR baseline."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset_seed", type=int, required=True)
    parser.add_argument(
        "--regression_loss",
        choices=REGRESSION_LOSS_KINDS,
        default="mse",
    )
    parser.add_argument("--mse_loss_weight", type=float, default=1.0)
    parser.add_argument("--wasserstein_loss_weight", type=float, default=0.0)
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
        "train_batch_size",
        "num_train_iterations",
        "gradient_accumulation_batches",
        "log_every_iterations",
        "eval_every_iterations",
        "num_fixed_eval_samples",
        "eval_batch_size",
        "eval_metric_num_quantiles",
        "checkpoint_every_iterations",
        "lr_num_cycles",
    ):
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
    for name in ("train_dataset_seed", "fixed_eval_dataset_seed", "seed"):
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
    if not math.isfinite(args.mse_loss_weight) or args.mse_loss_weight <= 0.0:
        raise ValueError("mse_loss_weight must be finite and positive.")
    if (
        not math.isfinite(args.wasserstein_loss_weight)
        or args.wasserstein_loss_weight < 0.0
    ):
        raise ValueError("wasserstein_loss_weight must be finite and nonnegative.")
    if args.regression_loss == "mse" and args.wasserstein_loss_weight != 0.0:
        raise ValueError("The MSE baseline requires wasserstein_loss_weight=0.")
    if args.regression_loss == "mse_w1" and args.wasserstein_loss_weight <= 0.0:
        raise ValueError(
            "The MSE+W1 baseline requires a positive wasserstein_loss_weight."
        )
    validate_continuous_dataset_arguments(args)
    if args.num_fixed_eval_samples < 16:
        raise ValueError(
            "num_fixed_eval_samples must be at least 16 for the 4x4 "
            "reconstruction grid."
        )
    if not 0.0 <= args.eval_peak_relative_threshold <= 1.0:
        raise ValueError("eval_peak_relative_threshold must lie in [0,1].")
    if args.learning_rate <= 0.0 or args.adam_epsilon <= 0.0:
        raise ValueError("learning_rate and adam_epsilon must be positive.")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must lie in [0,1).")
    if args.adam_weight_decay < 0.0 or args.max_grad_norm <= 0.0:
        raise ValueError("Weight decay must be nonnegative and grad norm positive.")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must lie in (0,1).")
    if not 0.0 <= args.ema_min_decay <= args.ema_decay:
        raise ValueError("ema_min_decay must lie in [0,ema_decay].")
    if args.ema_inv_gamma <= 0.0 or args.ema_power <= 0.0:
        raise ValueError("EMA warmup values must be positive.")
    if not args.output_dir.strip() or not args.resume_from_checkpoint.strip():
        raise ValueError("Output and resume arguments must not be empty.")


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


def _resume_value_matches(current: Any, saved: Any) -> bool:
    """Compare serialized config trees while tolerating harmless float drift."""

    if isinstance(current, bool) or isinstance(saved, bool):
        return type(current) is type(saved) and current == saved
    if isinstance(current, (int, float)) and isinstance(saved, (int, float)):
        if not math.isfinite(float(current)) or not math.isfinite(float(saved)):
            return current == saved
        return math.isclose(
            float(current),
            float(saved),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    if isinstance(current, dict) and isinstance(saved, dict):
        return current.keys() == saved.keys() and all(
            _resume_value_matches(current[key], saved[key]) for key in current
        )
    if isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
        return len(current) == len(saved) and all(
            _resume_value_matches(left, right) for left, right in zip(current, saved)
        )
    return current == saved


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
                    "Checkpoint artifacts exist, but none is a complete "
                    "resumable state."
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
    # Checkpoints produced before the explicit loss contract are exactly the
    # MSE defaults.  Materialize those defaults only for backward-compatible
    # resume; MSE+W1 checkpoints must always store their full contract.
    saved = dict(saved)
    if (
        saved.get("experiment") == EXPERIMENT_NAME
        or state.get("experiment") == EXPERIMENT_NAME
    ):
        saved.setdefault("regression_loss", "mse")
        saved.setdefault("mse_loss_weight", 1.0)
        saved.setdefault("wasserstein_loss_weight", 0.0)
        saved.setdefault(
            "regression_loss_contract",
            {
                "kind": "mse",
                "mse_loss_weight": 1.0,
                "wasserstein_loss_weight": 0.0,
                "wasserstein_ground_axis": "natural_log_relaxation_rate",
            },
        )
    mismatches = [
        f"{name}: saved={saved.get(name)!r}, current={value!r}"
        for name, value in current.items()
        if name not in mutable and not _resume_value_matches(value, saved.get(name))
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
    global_iteration = state.get("global_iteration")
    if (
        isinstance(global_iteration, bool)
        or not isinstance(global_iteration, int)
        or global_iteration < 0
    ):
        raise ValueError("Checkpoint global_iteration must be a nonnegative integer.")
    if global_iteration > requested_iterations:
        raise ValueError(
            "Checkpoint global_iteration exceeds the requested training horizon: "
            f"checkpoint={global_iteration}, requested={requested_iterations}."
        )
    return global_iteration


def _gather_fixed_predictions(
    args: argparse.Namespace,
    accelerator: Accelerator,
    model,
    loader,
    dataset,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    sampler = DeterministicRegressionSampler(model, dataset.log_rate_edges)
    prediction_batches = []
    reference_batches = []
    weight_batches = []
    target_batches = []
    measurement_batches = []
    started = start_distributed_wall_timer(accelerator)
    for batch in loader:
        measurements = batch["measurement"]
        condition = make_condition(
            measurements,
            dataset.echo_times,
            output_size=args.num_spectrum_bins,
            noise_standard_deviation=args.noise_standard_deviation,
        )
        predictions = sampler.predict(condition)[:, None, :]
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
            prediction, reference, weight, target, measurement = gathered
            prediction_batches.append(prediction.detach().cpu())
            reference_batches.append(reference.detach().cpu())
            weight_batches.append(weight.detach().cpu())
            target_batches.append(target.detach().cpu())
            measurement_batches.append(measurement.detach().cpu())
    elapsed = slowest_distributed_elapsed_seconds(accelerator, started)
    if not accelerator.is_main_process:
        empty = torch.empty(0)
        return empty, empty, empty, empty, empty, elapsed
    return (
        torch.cat(prediction_batches),
        torch.cat(reference_batches),
        torch.cat(weight_batches),
        torch.cat(target_batches),
        torch.cat(measurement_batches),
        elapsed,
    )


def _save_fixed_reconstruction_artifacts(
    evaluation_dir: Path,
    *,
    iteration: int,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    measurements: torch.Tensor,
    dataset,
    noise_standard_deviation: float,
) -> None:
    """Save the same spectrum/decay and 4x4 artifacts used by FM trainers."""

    figure_title = f"Deterministic regression | iteration {iteration}"
    qualitative_samples = predictions[0]
    save_reconstruction_figure(
        evaluation_dir / "latest_reconstruction.png",
        qualitative_samples,
        targets[0],
        measurements[0],
        dataset.rate_centers,
        dataset.echo_times,
        dataset.kernel,
        noise_standard_deviation=noise_standard_deviation,
        figure_title=figure_title,
    )

    grid_samples = predictions[:16]
    grid_targets = targets[:16]
    save_reconstruction_grid_figure(
        evaluation_dir / "latest_reconstruction_grid.png",
        grid_samples,
        grid_targets,
        dataset.rate_centers,
    )


def _compute_fixed_evaluation_metrics(
    predictions: torch.Tensor,
    references: torch.Tensor,
    weights: torch.Tensor,
    targets: torch.Tensor,
    measurements: torch.Tensor,
    dataset,
    *,
    noise_standard_deviation: float,
    w2_num_quantiles: int,
    peak_threshold_fraction: float,
) -> dict[str, object]:
    """Low-cost point validation; posterior-set evaluation remains offline."""

    del references, weights
    result = evaluate_histogram_samples(
        predictions,
        targets,
        measurements,
        dataset.kernel,
        dataset.log_rate_edges,
        noise_standard_deviation=noise_standard_deviation,
        w2_num_quantiles=w2_num_quantiles,
        peak_threshold_fraction=peak_threshold_fraction,
    )
    result["fixed_validation_metric_protocol"] = FIXED_VALIDATION_METRIC_PROTOCOL
    return result


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
    try:
        if ema is not None:
            ema.store(unwrapped.parameters())
            ema.copy_to(unwrapped.parameters())
            ema_stored = True
        model.eval()
        (
            predictions,
            references,
            weights,
            targets,
            measurements,
            elapsed,
        ) = _gather_fixed_predictions(args, accelerator, model, loader, dataset)
    finally:
        if ema_stored:
            ema.restore(unwrapped.parameters())
        model.train(was_training)
    postprocess_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            row = _compute_fixed_evaluation_metrics(
                predictions,
                references,
                weights,
                targets,
                measurements,
                dataset,
                noise_standard_deviation=args.noise_standard_deviation,
                w2_num_quantiles=args.eval_metric_num_quantiles,
                peak_threshold_fraction=args.eval_peak_relative_threshold,
            )
            record = {
                "iteration": int(iteration),
                "model_weight": "ema" if ema else "raw",
                **row,
                "num_conditions": int(predictions.shape[0]),
                "samples_per_condition": 1,
                "unique_model_predictions_per_condition": 1,
                "actual_model_nfe_per_condition": 1,
                "inference_wall_seconds": float(elapsed),
                "inference_seconds_per_generated_sample": float(elapsed)
                / float(predictions.shape[0]),
                "timing_interpretation": (
                    "maximum_rank_wall_time_divided_by_global_condition_count; "
                    "includes the evaluation data loop and distributed tensor "
                    "gather; throughput_equivalent_not_single_condition_latency"
                ),
            }
            evaluation_dir = output_dir / "eval" / "fixed_validation"
            metrics_history_path = evaluation_dir / "metrics.jsonl"
            append_jsonl(metrics_history_path, record)
            write_json(evaluation_dir / "latest_metrics.json", record)
            _save_fixed_reconstruction_artifacts(
                evaluation_dir,
                iteration=iteration,
                predictions=predictions,
                targets=targets,
                measurements=measurements,
                dataset=dataset,
                noise_standard_deviation=args.noise_standard_deviation,
            )
            print(
                format_console_log_record(record, indent=2, sort_keys=True),
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
        accelerator,
        postprocess_error,
        "deterministic fixed-evaluation postprocessing",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_batches,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)
    global_eval_batch = args.eval_batch_size * accelerator.num_processes
    if args.num_fixed_eval_samples % global_eval_batch:
        raise ValueError(
            "num_fixed_eval_samples must be divisible by eval_batch_size * world size."
        )
    output_dir = Path(args.output_dir)
    output_setup_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            output_setup_error = error
    _collective_raise_if_failed(
        accelerator, output_setup_error, "deterministic output-directory setup"
    )

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
            model_cls=UNet1DModel,
            model_config=unwrapped.config,
        )
        ema.to(accelerator.device)
        accelerator.register_for_checkpointing(ema)
    trainer = DeterministicRegressionTrainer(
        fixed_dataset.log_rate_edges,
        loss_kind=args.regression_loss,
        mse_loss_weight=args.mse_loss_weight,
        wasserstein_loss_weight=args.wasserstein_loss_weight,
    )

    config = dict(vars(args))
    config.update(
        experiment=_experiment_name(args),
        **continuous_dataset_config_values(args),
        training_stream_generation={
            "master_seed": int(args.train_dataset_seed),
            "split": "train",
            "infinite_on_the_fly": True,
            "global_pair_ids_are_iteration_addressed": True,
        },
        network_architecture=get_network_architecture_signature(
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
        ),
        deterministic_input_representation=(
            DETERMINISTIC_REGRESSION_INPUT_REPRESENTATION
        ),
        deterministic_output_parameterization=(
            DETERMINISTIC_REGRESSION_OUTPUT_PARAMETERIZATION
        ),
        deterministic_timestep=DETERMINISTIC_REGRESSION_TIMESTEP,
        regression_loss_contract={
            "kind": args.regression_loss,
            "mse_loss_weight": args.mse_loss_weight,
            "wasserstein_loss_weight": args.wasserstein_loss_weight,
            "wasserstein_ground_axis": "natural_log_relaxation_rate",
        },
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

    resume = None
    resume_resolution_error: Exception | None = None
    try:
        resume = _resolve_resume(args)
    except Exception as error:
        resume_resolution_error = error
    _collective_raise_if_failed(
        accelerator,
        resume_resolution_error,
        "deterministic checkpoint resolution",
    )
    global_iteration = 0
    resume_validation_error: Exception | None = None
    if resume is not None:
        try:
            state = _read_state(resume)
            if state.get("experiment") != _experiment_name(args):
                raise ValueError(
                    "Checkpoint experiment does not match deterministic regression."
                )
            _validate_resume_config(config, state)
            global_iteration = _resume_progress(
                state,
                requested_iterations=args.num_train_iterations,
            )
        except Exception as error:
            resume_validation_error = error
    _collective_raise_if_failed(
        accelerator,
        resume_validation_error,
        "deterministic checkpoint metadata validation",
    )
    resume_load_error: Exception | None = None
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
        "deterministic checkpoint state loading",
    )
    if resume is not None:
        accelerator.print(
            f"Resumed deterministic-regression training from {resume} "
            f"after iteration {global_iteration}."
        )
    history_setup_error: Exception | None = None
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
                discard_latest_images=(resume is not None),
            )
            save_training_curves(
                output_dir / "loss_history.jsonl",
                output_dir / "eval" / "fixed_validation" / "metrics.jsonl",
                output_dir / "eval" / "loss" / "curve_loss.png",
            )
        except Exception as error:
            history_setup_error = error
    _collective_raise_if_failed(
        accelerator, history_setup_error, "deterministic history setup"
    )

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
    normalized_sum = torch.zeros_like(loss_sum)
    log_w1_sum = torch.zeros_like(loss_sum)
    mse_component_sum = torch.zeros_like(loss_sum)
    wasserstein_component_sum = torch.zeros_like(loss_sum)
    batch_count = torch.zeros_like(loss_sum)
    skipped_optimizer_updates = 0
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
                normalized_sum += (
                    logs["normalized_histogram_mse"].detach().to(torch.float64)
                )
                log_w1_sum += logs["histogram_log_w1"].detach().to(torch.float64)
                mse_component_sum += (
                    logs["weighted_mse_loss_component"].detach().to(torch.float64)
                )
                wasserstein_component_sum += (
                    logs["weighted_wasserstein_loss_component"]
                    .detach()
                    .to(torch.float64)
                )
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
                step_was_skipped = accelerator.optimizer_step_was_skipped
                if step_was_skipped:
                    skipped_optimizer_updates += 1
                if ema is not None and not step_was_skipped:
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
                torch.stack(
                    (
                        loss_sum,
                        normalized_sum,
                        log_w1_sum,
                        mse_component_sum,
                        wasserstein_component_sum,
                        batch_count,
                    )
                )[None, :]
            ).sum(dim=0)
            logging_interval_wall_seconds = slowest_distributed_elapsed_seconds(
                accelerator, logging_interval_started
            )
            run_elapsed_wall_seconds = slowest_distributed_elapsed_seconds(
                accelerator, run_started
            )
            logging_error: Exception | None = None
            if accelerator.is_main_process:
                try:
                    denominator = float(gathered[5].item())
                    record = {
                        "iteration": iteration,
                        "loss": float(gathered[0].item()) / denominator,
                        "normalized_histogram_mse": (
                            float(gathered[1].item()) / denominator
                        ),
                        "histogram_log_w1": float(gathered[2].item()) / denominator,
                        "weighted_mse_loss_component": (
                            float(gathered[3].item()) / denominator
                        ),
                        "weighted_wasserstein_loss_component": (
                            float(gathered[4].item()) / denominator
                        ),
                        "microbatch_count": int(gathered[5].item()),
                        "skipped_optimizer_updates": int(skipped_optimizer_updates),
                        "logging_interval_wall_seconds": (
                            logging_interval_wall_seconds
                        ),
                        "run_elapsed_wall_seconds": run_elapsed_wall_seconds,
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
                accelerator, logging_error, "deterministic iteration logging"
            )
            loss_sum.zero_()
            normalized_sum.zero_()
            log_w1_sum.zero_()
            mse_component_sum.zero_()
            wasserstein_component_sum.zero_()
            batch_count.zero_()
            skipped_optimizer_updates = 0
            logging_interval_started = start_distributed_wall_timer(accelerator)
        should_evaluate = (
            iteration % args.eval_every_iterations == 0 or is_final_iteration
        )
        if should_evaluate:
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
        should_checkpoint = (
            iteration % args.checkpoint_every_iterations == 0 or is_final_iteration
        )
        if should_checkpoint:
            checkpoint = save_latest_accelerator_checkpoint(
                accelerator,
                model,
                ema,
                output_dir,
                iteration=global_iteration,
                training_config=config,
                experiment=_experiment_name(args),
            )
            accelerator.print(
                "Saved deterministic-regression checkpoint after iteration "
                f"{iteration}: {checkpoint}"
            )
    accelerator.print(
        "Deterministic-regression training finished after iteration "
        f"{global_iteration}."
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
