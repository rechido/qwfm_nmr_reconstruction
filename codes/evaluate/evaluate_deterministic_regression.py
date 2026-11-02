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
from diffusers import UNet1DModel
from torch.utils.data import DataLoader


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from samplers.deterministic_regression_sampler import DeterministicRegressionSampler
from utils.checkpoint_utils import validate_checkpoint_path
from utils.deterministic_regression_utils import (
    DETERMINISTIC_REGRESSION_INPUT_REPRESENTATION,
    DETERMINISTIC_REGRESSION_OUTPUT_PARAMETERIZATION,
    DETERMINISTIC_REGRESSION_TIMESTEP,
    REGRESSION_LOSS_KINDS,
)
from utils.continuous_experiment_utils import (
    add_continuous_dataset_arguments,
    continuous_dataset_config_values,
    spec_from_args,
    validate_continuous_dataset_arguments,
)
from utils.nmr_utils import (
    find_latest_checkpoint,
    make_condition,
    save_reconstruction_figure,
    save_reconstruction_grid_figure,
    write_json,
)
from utils.unet_config import (
    get_network_architecture_signature,
    validate_unet_config,
)
from utils.training_utils import format_console_log_record
from utils.posterior_evaluation_utils import (
    PosteriorConditionDataset,
    build_sampling_artifact,
    evaluate_and_write_metrics,
    load_and_validate_references,
    load_reusable_sampling_artifact,
    parse_bool,
    save_or_reuse_sampling_artifact,
)
from evaluate.evaluate_saved_samples import EvaluationSettings


EXPERIMENT_NAME = "deterministic_regression"
MSE_W1_EXPERIMENT_NAME = "deterministic_regression_mse_w1"


def _experiment_name(args: argparse.Namespace) -> str:
    return (
        MSE_W1_EXPERIMENT_NAME if args.regression_loss == "mse_w1" else EXPERIMENT_NAME
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate matched-backbone deterministic NMR regression."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--model_subfolder", choices=("unet", "unet_ema"), required=True
    )
    parser.add_argument("--eval_output_subdir", type=str, required=True)
    parser.add_argument("--primary_reference", type=str, required=True)
    parser.add_argument("--oracle_reference", type=str, required=True)
    add_continuous_dataset_arguments(parser)
    parser.add_argument(
        "--regression_loss",
        choices=REGRESSION_LOSS_KINDS,
        required=True,
    )
    parser.add_argument("--mse_loss_weight", type=float, required=True)
    parser.add_argument("--wasserstein_loss_weight", type=float, required=True)
    parser.add_argument("--eval_batch_size", type=int, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, required=True)
    parser.add_argument("--num_posterior_samples", type=int, required=True)
    parser.add_argument("--num_visual_samples", type=int, required=True)
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
    parser.add_argument(
        "--mixed_precision", choices=("no", "fp16", "bf16"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    bandwidth = getattr(args, "kernel_bandwidth", None)
    if bandwidth is not None and (not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("kernel_bandwidth must be finite and positive.")
    for name in (
        "num_measurements",
        "num_spectrum_bins",
        "eval_batch_size",
        "num_posterior_samples",
        "num_visual_samples",
        "reference_repeats",
        "nearest_k",
        "w2_num_quantiles",
        "distance_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.num_spectrum_bins < 8:
        raise ValueError("num_spectrum_bins must be at least eight.")
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be nonnegative.")
    if not 0 <= args.metric_seed < 2**63 or not 0 <= args.seed < 2**63:
        raise ValueError("Seeds must lie in [0,2**63).")
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
    if args.num_visual_samples > args.num_posterior_samples:
        raise ValueError("num_visual_samples cannot exceed num_posterior_samples.")
    if args.num_posterior_samples <= args.nearest_k:
        raise ValueError("num_posterior_samples must exceed nearest_k for PRDC.")
    if not 0.0 < args.credible_interval_level < 1.0:
        raise ValueError("credible_interval_level must lie in (0,1).")
    if not 0.0 <= args.peak_relative_threshold <= 1.0:
        raise ValueError("peak_relative_threshold must lie in [0,1].")
    relative = Path(args.eval_output_subdir)
    if (
        not args.output_dir.strip()
        or not args.checkpoint_path.strip()
        or not args.eval_output_subdir.strip()
        or not args.primary_reference.strip()
        or not args.oracle_reference.strip()
        or relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
    ):
        raise ValueError("Output/checkpoint/evaluation paths are invalid.")


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_path.strip().lower() == "latest":
        checkpoint = find_latest_checkpoint(
            Path(args.output_dir),
            require_ema=args.model_subfolder == "unet_ema",
        )
    else:
        checkpoint = validate_checkpoint_path(
            Path(args.checkpoint_path).expanduser(),
            require_ema=args.model_subfolder == "unet_ema",
            allow_legacy_numbered=False,
        )
    if not (checkpoint / args.model_subfolder / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint model: {checkpoint}")
    return checkpoint


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _checkpoint_iteration(state: dict[str, Any]) -> int:
    value = state.get("global_iteration")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Checkpoint global_iteration must be nonnegative.")
    return int(value)


def _validate_checkpoint(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    root = _read_json_object(Path(args.output_dir) / "training_config.json")
    state = _read_json_object(checkpoint / "training_state.json")
    expected = {
        "experiment": _experiment_name(args),
        **continuous_dataset_config_values(args),
        "network_architecture": get_network_architecture_signature(
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
        ),
        "deterministic_input_representation": (
            DETERMINISTIC_REGRESSION_INPUT_REPRESENTATION
        ),
        "deterministic_output_parameterization": (
            DETERMINISTIC_REGRESSION_OUTPUT_PARAMETERIZATION
        ),
        "deterministic_timestep": DETERMINISTIC_REGRESSION_TIMESTEP,
        "regression_loss_contract": {
            "kind": args.regression_loss,
            "mse_loss_weight": args.mse_loss_weight,
            "wasserstein_loss_weight": args.wasserstein_loss_weight,
            "wasserstein_ground_axis": "natural_log_relaxation_rate",
        },
    }
    for label, config in (
        ("root training_config", root),
        ("checkpoint training_config", state.get("training_config")),
    ):
        if not isinstance(config, dict):
            raise ValueError(f"{label} is missing.")
        config = dict(config)
        mismatches = []
        for name, expected_value in expected.items():
            actual = config.get(name)
            if isinstance(expected_value, float):
                matched = isinstance(actual, (int, float)) and math.isclose(
                    float(actual), expected_value, rel_tol=0.0, abs_tol=1.0e-12
                )
            else:
                matched = actual == expected_value
            if not matched:
                mismatches.append(
                    f"{name}: saved={actual!r}, evaluation={expected_value!r}"
                )
        if mismatches:
            raise ValueError(f"{label} mismatch:\n" + "\n".join(mismatches))
    if state.get("experiment") != _experiment_name(args):
        raise ValueError("Checkpoint experiment mismatch.")
    _checkpoint_iteration(state)
    return state


def _collective_raise_if_failed(
    accelerator: Accelerator,
    local_error: Exception | None,
    phase: str,
) -> None:
    failure = torch.tensor(
        int(local_error is not None), device=accelerator.device, dtype=torch.int64
    )
    failure_count = int(accelerator.reduce(failure, reduction="sum").item())
    if failure_count:
        if local_error is None:
            raise RuntimeError(f"{phase} failed on {failure_count} other process(es).")
        raise RuntimeError(f"{phase} failed: {local_error}") from local_error


def _expected_artifact_provenance(
    args: argparse.Namespace,
    *,
    spec,
    primary_reference: dict[str, Any],
    training_config: dict[str, Any],
    checkpoint: Path,
    world_size: int,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 2,
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "condition_ids": primary_reference["condition_ids"],
        "measurements": primary_reference["measurements"],
        "grid": primary_reference["grid"],
        "method": _experiment_name(args),
        "training_config": training_config,
        "training_output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "weights": "ema" if args.model_subfolder == "unet_ema" else "raw",
        "model_subfolder": args.model_subfolder,
        "sampling_config": {
            "sampler": "deterministic_regression",
            "regression_loss": args.regression_loss,
            "mse_loss_weight": args.mse_loss_weight,
            "wasserstein_loss_weight": args.wasserstein_loss_weight,
            "materialization": "repeat_one_unique_prediction",
        },
        "inference_settings": {
            "mixed_precision": args.mixed_precision,
            "world_size": int(world_size),
            "eval_batch_size": int(args.eval_batch_size),
        },
        "source_seed": int(args.seed),
        "num_samples_per_condition": int(args.num_posterior_samples),
        "state_size": int(args.num_spectrum_bins),
        "deterministic": True,
        "reference_path": str(Path(args.primary_reference).expanduser().resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed, device_specific=True)
    checkpoint = _resolve_checkpoint(args)
    checkpoint_state = _validate_checkpoint(args, checkpoint)
    training_config = _read_json_object(Path(args.output_dir) / "training_config.json")
    spec = spec_from_args(args)
    primary_reference, oracle_reference = load_and_validate_references(
        args.primary_reference, args.oracle_reference, spec
    )
    dataset = PosteriorConditionDataset(primary_reference)
    global_batch = args.eval_batch_size * accelerator.num_processes
    if len(dataset) % global_batch:
        raise ValueError(
            "Reference condition count must be divisible by eval_batch_size * world size."
        )
    evaluation_dir = Path(args.output_dir) / args.eval_output_subdir
    samples_path = evaluation_dir / "samples.pt"
    provenance = _expected_artifact_provenance(
        args,
        spec=spec,
        primary_reference=primary_reference,
        training_config=training_config,
        checkpoint=checkpoint,
        world_size=accelerator.num_processes,
    )
    if args.reuse_existing:
        reuse_error = None
        artifact = None
        try:
            artifact = load_reusable_sampling_artifact(
                samples_path,
                expected_provenance=provenance,
                primary_reference=primary_reference,
                oracle_reference=oracle_reference,
                sample_count=args.num_posterior_samples,
            )
        except Exception as error:
            reuse_error = error
        _collective_raise_if_failed(
            accelerator, reuse_error, "cached-sample validation"
        )
        postprocess_error = None
        if accelerator.is_main_process:
            try:
                assert artifact is not None
                evaluate_and_write_metrics(
                    artifact,
                    primary_reference,
                    oracle_reference,
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
                    device=accelerator.device,
                )
            except Exception as error:
                postprocess_error = error
        _collective_raise_if_failed(
            accelerator, postprocess_error, "cached metric postprocessing"
        )
        accelerator.end_training()
        return
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    model = UNet1DModel.from_pretrained(checkpoint / args.model_subfolder)
    validate_unet_config(
        model,
        sample_size=args.num_spectrum_bins,
        num_measurements=args.num_measurements,
    )
    model, loader = accelerator.prepare(model, loader)
    model.eval()
    sampler = DeterministicRegressionSampler(model, dataset.log_rate_edges)
    prediction_batches = []
    measurement_batches = []
    target_batches = []
    condition_id_batches = []
    accelerator.wait_for_everyone()
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    started = time.perf_counter()
    for batch in loader:
        measurements = batch["measurement"]
        condition = make_condition(
            measurements,
            dataset.echo_times,
            output_size=args.num_spectrum_bins,
            noise_standard_deviation=args.noise_standard_deviation,
        )
        unique_predictions = sampler.predict(condition)
        gathered = accelerator.gather_for_metrics(
            (
                unique_predictions, measurements, batch["target_histogram"], batch["condition_id"],
            )
        )
        if accelerator.is_main_process:
            prediction, measurement, target, condition_id = gathered
            target_batches.append(target.detach().cpu())
            prediction_batches.append(prediction.detach().cpu())
            measurement_batches.append(measurement.detach().cpu())
            condition_id_batches.append(condition_id.detach().cpu())
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    local_elapsed = torch.tensor(
        [time.perf_counter() - started],
        device=accelerator.device,
        dtype=torch.float64,
    )
    elapsed = float(accelerator.gather(local_elapsed).max().item())

    postprocess_error = None
    if accelerator.is_main_process:
        try:
            unique_predictions = torch.cat(prediction_batches)
            measurements = torch.cat(measurement_batches)
            targets = torch.cat(target_batches)
            condition_ids = torch.cat(condition_id_batches)
            order = torch.argsort(condition_ids)
            unique_predictions = unique_predictions[order]
            measurements = measurements[order]
            targets = targets[order]
            condition_ids = condition_ids[order]
            if not torch.equal(condition_ids, primary_reference["condition_ids"].cpu()):
                raise RuntimeError(
                    "Gathered condition IDs differ from the SMC reference."
                )
            materialized_predictions = (
                unique_predictions[:, None]
                .expand(-1, args.num_posterior_samples, -1)
                .clone()
            )
            artifact = build_sampling_artifact(
                spec=spec,
                primary_reference=primary_reference,
                samples=materialized_predictions,
                method=_experiment_name(args),
                training_config=training_config,
                output_dir=args.output_dir,
                checkpoint=checkpoint,
                model_subfolder=args.model_subfolder,
                primary_reference_path=args.primary_reference,
                state_size=args.num_spectrum_bins,
                deterministic=True,
                sampling_config=provenance["sampling_config"],
                source_seed=args.seed,
                sampling_wall_seconds=elapsed,
                inference_settings=provenance["inference_settings"],
            )
            artifact = save_or_reuse_sampling_artifact(
                samples_path,
                artifact,
                reuse_existing=False,
                primary_reference=primary_reference,
                oracle_reference=oracle_reference,
                sample_count=args.num_posterior_samples,
            )
            evaluate_and_write_metrics(
                artifact,
                primary_reference,
                oracle_reference,
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
                device=accelerator.device,
            )
            write_json(
                evaluation_dir / "sampling_summary.json",
                {
                    "experiment": _experiment_name(args),
                    "checkpoint_iteration": _checkpoint_iteration(checkpoint_state),
                    "num_conditions": len(dataset),
                    "num_posterior_samples": args.num_posterior_samples,
                    "unique_model_predictions_per_condition": 1,
                    "sampling_wall_seconds": elapsed,
                },
            )
            save_reconstruction_figure(
                evaluation_dir / "ground_truth_vs_reconstruction.png",
                materialized_predictions[0, : args.num_visual_samples],
                targets[0],
                measurements[0],
                dataset.rate_centers,
                dataset.echo_times,
                dataset.kernel,
                args.noise_standard_deviation,
                figure_title="Deterministic regression: one unique prediction",
            )
            if len(dataset) >= 16:
                save_reconstruction_grid_figure(
                    evaluation_dir / "ground_truth_vs_reconstruction_grid.png",
                    materialized_predictions[:16],
                    targets[:16],
                    dataset.rate_centers,
                )
            print(
                format_console_log_record(
                    {"sampling_wall_seconds": elapsed}, sort_keys=True
                ),
                flush=True,
            )
        except Exception as error:
            postprocess_error = error
    _collective_raise_if_failed(
        accelerator, postprocess_error, "deterministic evaluation postprocessing"
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
