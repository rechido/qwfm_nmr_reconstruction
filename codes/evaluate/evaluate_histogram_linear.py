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

from samplers.histogram_linear_sampler import HistogramLinearSampler
from utils.checkpoint_utils import validate_checkpoint_path
from utils.nmr_utils import (
    find_latest_checkpoint,
    make_condition,
    save_reconstruction_figure,
    save_reconstruction_grid_figure,
    write_json,
)
from utils.continuous_experiment_utils import (
    add_continuous_dataset_arguments,
    continuous_dataset_config_values,
    spec_from_args,
    validate_continuous_dataset_arguments,
)
from utils.posterior_evaluation_utils import (
    PosteriorConditionDataset,
    build_sampling_artifact,
    evaluate_and_write_metrics,
    load_and_validate_references,
    load_reusable_sampling_artifact,
    parse_bool,
    save_or_reuse_sampling_artifact,
)
from utils.unet_config import (
    get_network_architecture_signature,
    validate_unet_config,
)
from utils.training_utils import format_console_log_record
from evaluate.evaluate_saved_samples import EvaluationSettings


EXPERIMENT_NAME = "histogram_linear"


def _parse_positive_integer_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of positive integers."
        ) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "Every evaluation ODE-step count must be positive."
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "Evaluation ODE-step counts must not contain duplicates."
        )
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate histogram-linear NMR flow matching from a checkpoint."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--model_subfolder",
        choices=("unet", "unet_ema"),
        required=True,
    )
    parser.add_argument("--eval_output_subdir", type=str, required=True)
    parser.add_argument("--primary_reference", type=str, required=True)
    parser.add_argument("--oracle_reference", type=str, required=True)
    add_continuous_dataset_arguments(parser)
    parser.add_argument("--eval_batch_size", type=int, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, required=True)
    parser.add_argument("--num_posterior_samples", type=int, required=True)
    parser.add_argument(
        "--eval_ode_steps",
        type=_parse_positive_integer_list,
        required=True,
    )
    parser.add_argument("--num_visual_samples", type=int, required=True)
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source_seed", type=int, required=True)
    parser.add_argument("--model_time_scale", type=int, required=True)
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
        "model_time_scale",
        "reference_repeats",
        "nearest_k",
        "w2_num_quantiles",
        "distance_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    validate_continuous_dataset_arguments(args)
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be nonnegative.")
    if args.num_visual_samples > args.num_posterior_samples:
        raise ValueError("num_visual_samples cannot exceed num_posterior_samples.")
    if args.num_posterior_samples <= args.nearest_k:
        raise ValueError("num_posterior_samples must exceed nearest_k for PRDC.")
    for name in ("metric_seed", "seed", "source_seed"):
        value = int(getattr(args, name))
        if value < 0 or value >= 2**63:
            raise ValueError(f"{name} must lie in [0, 2**63).")
    if not math.isfinite(args.credible_interval_level) or not (
        0.0 < args.credible_interval_level < 1.0
    ):
        raise ValueError("credible_interval_level must lie in (0,1).")
    if not 0.0 <= args.peak_relative_threshold <= 1.0:
        raise ValueError("peak_relative_threshold must lie in [0, 1].")
    for name in ("output_dir", "checkpoint_path", "primary_reference", "oracle_reference"):
        if not str(getattr(args, name)).strip():
            raise ValueError(f"{name} must not be empty.")
    if Path(args.primary_reference).expanduser().resolve() == Path(
        args.oracle_reference
    ).expanduser().resolve():
        raise ValueError("primary_reference and oracle_reference must differ.")
    relative_output = Path(args.eval_output_subdir)
    if (
        not args.eval_output_subdir.strip()
        or relative_output.is_absolute()
        or relative_output == Path(".")
        or ".." in relative_output.parts
    ):
        raise ValueError(
            "eval_output_subdir must be a nonempty relative path without '..'."
        )


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser()
    if args.checkpoint_path.strip().lower() == "latest":
        checkpoint = find_latest_checkpoint(
            output_dir,
            require_ema=args.model_subfolder == "unet_ema",
        )
    else:
        checkpoint = Path(args.checkpoint_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        checkpoint = validate_checkpoint_path(
            checkpoint,
            require_ema=args.model_subfolder == "unet_ema",
            allow_legacy_numbered=False,
        )
    model_directory = checkpoint / args.model_subfolder
    if not (model_directory / "config.json").is_file():
        raise FileNotFoundError(
            f"Checkpoint model subfolder is incomplete: {model_directory}"
        )
    return checkpoint


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _expected_training_values(args: argparse.Namespace) -> dict[str, object]:
    return {
        "experiment": EXPERIMENT_NAME,
        **continuous_dataset_config_values(args),
        "network_architecture": get_network_architecture_signature(
            sample_size=args.num_spectrum_bins,
            num_measurements=args.num_measurements,
        ),
    }


def _matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
    return actual == expected


def _validate_config_mapping(
    label: str,
    config: dict[str, Any],
    expected: dict[str, object],
) -> None:
    mismatches = []
    for name, expected_value in expected.items():
        if name not in config:
            if name == "network_architecture":
                mismatches.append(
                    "network_architecture: missing; "
                    "cannot verify the stock UNet1DModel architecture"
                )
                continue
            mismatches.append(f"{name}: missing")
        elif not _matches(config[name], expected_value):
            mismatches.append(
                f"{name}: saved={config[name]!r}, evaluation={expected_value!r}"
            )
    if mismatches:
        raise ValueError(
            f"{label} is incompatible with this evaluation:\n" + "\n".join(mismatches)
        )


def _validate_saved_model_time_scale(
    label: str,
    config: dict[str, Any],
    expected: int,
) -> None:
    has_new_name = "model_time_scale" in config
    has_legacy_name = "fm_num_timesteps" in config
    if not has_new_name and not has_legacy_name:
        raise ValueError(
            f"{label} is incompatible with this evaluation:\n"
            "model_time_scale: missing (legacy fm_num_timesteps is also absent)"
        )
    if (
        has_new_name
        and has_legacy_name
        and not _matches(config["model_time_scale"], config["fm_num_timesteps"])
    ):
        raise ValueError(
            f"{label} contains conflicting model-time scales: "
            f"model_time_scale={config['model_time_scale']!r}, "
            f"fm_num_timesteps={config['fm_num_timesteps']!r}."
        )
    saved = config["model_time_scale"] if has_new_name else config["fm_num_timesteps"]
    if not _matches(saved, expected):
        raise ValueError(
            f"{label} is incompatible with this evaluation:\n"
            f"model_time_scale: saved={saved!r}, evaluation={expected!r}"
        )


def _checkpoint_global_iteration(checkpoint_state: dict[str, Any]) -> int:
    value = checkpoint_state.get("global_iteration")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Checkpoint global_iteration must be a nonnegative integer.")
    return value


def _validate_saved_training_configuration(
    args: argparse.Namespace,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _expected_training_values(args)
    root_config_path = Path(args.output_dir) / "training_config.json"
    root_config = _read_json_object(root_config_path)
    _validate_config_mapping("Root training_config.json", root_config, expected)
    _validate_saved_model_time_scale(
        "Root training_config.json",
        root_config,
        args.model_time_scale,
    )

    checkpoint_state = _read_json_object(checkpoint / "training_state.json")
    if checkpoint_state.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            "Checkpoint experiment mismatch: "
            f"expected {EXPERIMENT_NAME!r}, got {checkpoint_state.get('experiment')!r}."
        )
    checkpoint_config = checkpoint_state.get("training_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint metadata lacks a complete training_config.")
    _validate_config_mapping(
        "Checkpoint training_config",
        checkpoint_config,
        expected,
    )
    _validate_saved_model_time_scale(
        "Checkpoint training_config",
        checkpoint_config,
        args.model_time_scale,
    )
    _checkpoint_global_iteration(checkpoint_state)
    return root_config, checkpoint_state


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0.0:
        raise RuntimeError(f"Cannot compute {label}: denominator is nonpositive.")
    return numerator / denominator


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


def _expected_artifact_provenance(
    args: argparse.Namespace,
    *,
    spec,
    primary_reference: dict[str, Any],
    training_config: dict[str, Any],
    checkpoint: Path,
    num_ode_steps: int,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 2,
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "condition_ids": primary_reference["condition_ids"],
        "measurements": primary_reference["measurements"],
        "grid": primary_reference["grid"],
        "method": EXPERIMENT_NAME,
        "training_config": training_config,
        "training_output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "weights": "ema" if args.model_subfolder == "unet_ema" else "raw",
        "model_subfolder": args.model_subfolder,
        "sampling_config": {
            "sampler": "explicit_euler",
            "ode_steps": int(num_ode_steps),
            "model_time_scale": int(args.model_time_scale),
        },
        "inference_settings": {
            "mixed_precision": args.mixed_precision,
            "world_size": int(args._world_size),
            "eval_batch_size": int(args.eval_batch_size),
        },
        "source_seed": int(args.source_seed),
        "num_samples_per_condition": int(args.num_posterior_samples),
        "state_size": int(args.num_spectrum_bins),
        "deterministic": False,
        "reference_path": str(Path(args.primary_reference).expanduser().resolve()),
    }


def _evaluate_ode_steps(
    args: argparse.Namespace,
    accelerator: Accelerator,
    model,
    loader,
    dataset: PosteriorConditionDataset,
    checkpoint: Path,
    checkpoint_state: dict[str, Any],
    training_config: dict[str, Any],
    primary_reference: dict[str, Any],
    oracle_reference: dict[str, Any],
    num_ode_steps: int,
) -> None:
    spec = spec_from_args(args)
    nfe_directory = (
        Path(args.output_dir) / args.eval_output_subdir / f"nfe_{num_ode_steps:04d}"
    )
    samples_path = nfe_directory / "samples.pt"
    expected_provenance = _expected_artifact_provenance(
        args,
        spec=spec,
        primary_reference=primary_reference,
        training_config=training_config,
        checkpoint=checkpoint,
        num_ode_steps=num_ode_steps,
    )

    if args.reuse_existing:
        cache_error: Exception | None = None
        cached: dict[str, Any] | None = None
        if accelerator.is_main_process:
            try:
                cached = load_reusable_sampling_artifact(
                    samples_path,
                    expected_provenance=expected_provenance,
                    primary_reference=primary_reference,
                    oracle_reference=oracle_reference,
                    sample_count=args.num_posterior_samples,
                )
            except Exception as error:
                cache_error = error
        _collective_raise_if_failed(
            accelerator, cache_error, f"NFE={num_ode_steps} cache validation"
        )
        metric_error: Exception | None = None
        if accelerator.is_main_process:
            try:
                assert cached is not None
                evaluate_and_write_metrics(
                    cached,
                    primary_reference,
                    oracle_reference,
                    nfe_directory,
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
                metric_error = error
        _collective_raise_if_failed(
            accelerator, metric_error, f"NFE={num_ode_steps} cached metric evaluation"
        )
        accelerator.wait_for_everyone()
        return

    sampler = HistogramLinearSampler(
        model,
        accelerator.device,
        source_seed=args.source_seed + accelerator.process_index,
        model_time_scale=args.model_time_scale,
    )
    generated_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    measurement_batches: list[torch.Tensor] = []
    condition_id_batches: list[torch.Tensor] = []
    diagnostic_totals: dict[str, float] = {}

    accelerator.wait_for_everyone()
    _synchronize_device(accelerator.device)
    start_time = time.perf_counter()
    for batch in loader:
        targets = batch["target_histogram"]
        measurements = batch["measurement"]
        condition = make_condition(
            measurements,
            dataset.echo_times,
            output_size=args.num_spectrum_bins,
            noise_standard_deviation=args.noise_standard_deviation,
        )
        samples, diagnostics = sampler.sample(
            condition,
            num_samples_per_condition=args.num_posterior_samples,
            num_ode_steps=num_ode_steps,
        )
        gathered = accelerator.gather_for_metrics(
            (samples, targets, measurements, batch["condition_id"])
        )
        if accelerator.is_main_process:
            gathered_samples, gathered_targets, gathered_measurements, gathered_ids = gathered
            target_batches.append(gathered_targets.detach().cpu())
            generated_batches.append(gathered_samples.detach().cpu())
            measurement_batches.append(gathered_measurements.detach().cpu())
            condition_id_batches.append(gathered_ids.detach().cpu())
        for name, value in diagnostics.items():
            reduced = accelerator.reduce(value, reduction="sum")
            if accelerator.is_main_process:
                diagnostic_totals[name] = diagnostic_totals.get(name, 0.0) + float(
                    reduced.item()
                )
    _synchronize_device(accelerator.device)
    local_elapsed = torch.tensor(
        [time.perf_counter() - start_time],
        device=accelerator.device,
        dtype=torch.float64,
    )
    elapsed = float(accelerator.gather(local_elapsed).max().item())

    postprocess_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            if not generated_batches:
                raise RuntimeError("Evaluation loader produced no batches.")
            samples = torch.cat(generated_batches)
            targets = torch.cat(target_batches)
            measurements = torch.cat(measurement_batches)
            condition_ids = torch.cat(condition_id_batches)
            order = torch.argsort(condition_ids)
            samples = samples[order]
            targets = targets[order]
            measurements = measurements[order]
            condition_ids = condition_ids[order]
            if not torch.equal(condition_ids, primary_reference["condition_ids"].cpu()):
                raise RuntimeError("Gathered condition IDs differ from the SMC reference.")
            if not torch.allclose(
                measurements,
                primary_reference["measurements"].cpu(),
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise RuntimeError("Gathered measurements differ from the SMC reference.")

            artifact = build_sampling_artifact(
                spec=spec,
                primary_reference=primary_reference,
                samples=samples,
                method=EXPERIMENT_NAME,
                training_config=training_config,
                output_dir=args.output_dir,
                checkpoint=checkpoint,
                model_subfolder=args.model_subfolder,
                primary_reference_path=args.primary_reference,
                state_size=args.num_spectrum_bins,
                deterministic=False,
                sampling_config=expected_provenance["sampling_config"],
                source_seed=args.source_seed,
                sampling_wall_seconds=elapsed,
                inference_settings=expected_provenance["inference_settings"],
                sampler_diagnostic_totals=diagnostic_totals,
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
                nfe_directory,
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
                nfe_directory / "sampling_summary.json",
                {
                    "experiment": EXPERIMENT_NAME,
                    "checkpoint_iteration": _checkpoint_global_iteration(checkpoint_state),
                    "ode_steps": int(num_ode_steps),
                    "num_conditions": int(samples.shape[0]),
                    "num_posterior_samples": int(samples.shape[1]),
                    "sampling_wall_seconds": elapsed,
                    "sampler_diagnostic_totals": diagnostic_totals,
                },
            )
            save_reconstruction_figure(
                nfe_directory / "ground_truth_vs_reconstruction.png",
                samples[0, : args.num_visual_samples],
                targets[0],
                measurements[0],
                dataset.rate_centers,
                dataset.echo_times,
                dataset.kernel,
                noise_standard_deviation=args.noise_standard_deviation,
                figure_title=(
                    "Histogram-linear | checkpoint iteration "
                    f"{_checkpoint_global_iteration(checkpoint_state)} | "
                    f"Euler NFE {num_ode_steps}"
                ),
            )
            if samples.shape[0] >= 16:
                save_reconstruction_grid_figure(
                    nfe_directory / "ground_truth_vs_reconstruction_grid.png",
                    samples[:16],
                    targets[:16],
                    dataset.rate_centers,
                )
            print(
                format_console_log_record(
                    {
                        "nfe": int(num_ode_steps),
                        "num_conditions": int(samples.shape[0]),
                        "num_posterior_samples": int(samples.shape[1]),
                        "sampling_wall_seconds": elapsed,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:
            postprocess_error = error
    _collective_raise_if_failed(
        accelerator, postprocess_error, f"NFE={num_ode_steps} evaluation postprocessing"
    )
    accelerator.wait_for_everyone()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    args._world_size = accelerator.num_processes
    set_seed(args.seed, device_specific=True)
    if args.source_seed + accelerator.num_processes > 2**63:
        raise ValueError("source_seed plus process rank must lie below 2**63.")
    checkpoint = _resolve_checkpoint(args)
    root_config, checkpoint_state = _validate_saved_training_configuration(
        args,
        checkpoint,
    )
    reference_error: Exception | None = None
    primary_reference: dict[str, Any] | None = None
    oracle_reference: dict[str, Any] | None = None
    try:
        primary_reference, oracle_reference = load_and_validate_references(
            args.primary_reference,
            args.oracle_reference,
            spec_from_args(args),
        )
    except Exception as error:
        reference_error = error
    _collective_raise_if_failed(
        accelerator, reference_error, "posterior-reference validation"
    )
    assert primary_reference is not None and oracle_reference is not None
    dataset = PosteriorConditionDataset(primary_reference)
    global_eval_batch_size = args.eval_batch_size * accelerator.num_processes
    if len(dataset) % global_eval_batch_size:
        raise ValueError(
            "The reference condition count must be divisible by eval_batch_size "
            "times num_processes so rank-specific source streams remain stable."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    model = UNet1DModel.from_pretrained(
        checkpoint / args.model_subfolder,
    )
    validate_unet_config(
        model,
        sample_size=args.num_spectrum_bins,
        num_measurements=args.num_measurements,
    )
    model, loader = accelerator.prepare(model, loader)
    if len(loader) == 0:
        raise ValueError("The prepared evaluation loader has no batches.")
    model.eval()

    for num_ode_steps in args.eval_ode_steps:
        _evaluate_ode_steps(
            args,
            accelerator,
            model,
            loader,
            dataset,
            checkpoint,
            checkpoint_state,
            root_config,
            primary_reference,
            oracle_reference,
            num_ode_steps,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
