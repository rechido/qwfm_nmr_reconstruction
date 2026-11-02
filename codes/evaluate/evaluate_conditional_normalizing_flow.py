# ruff: noqa: E402
from __future__ import annotations
import argparse
import json
import math
import sys
from collections.abc import Mapping
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
from samplers.conditional_normalizing_flow_sampler import (
    ConditionalNormalizingFlowSampler,
)
from utils.conditional_normalizing_flow_utils import (
    CONDITIONAL_FLOW_INPUT_REPRESENTATION,
    CONDITIONAL_FLOW_OUTPUT_PARAMETERIZATION,
    CONDITIONAL_FLOW_TARGET_REPRESENTATION,
    get_conditional_flow_architecture_signature,
    load_conditional_flow_model_pretrained,
    validate_conditional_flow_model_config,
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
    write_json,
)
from utils.posterior_evaluation_utils import (
    SAMPLING_ARTIFACT_SCHEMA_VERSION,
    PosteriorConditionDataset,
    build_sampling_artifact,
    evaluate_and_write_metrics,
    load_and_validate_references,
    load_reusable_sampling_artifact,
    parse_bool,
    save_or_reuse_sampling_artifact,
)
from utils.training_utils import (
    nested_config_values_match,
    slowest_distributed_elapsed_seconds,
    start_distributed_wall_timer,
)

EXPERIMENT_NAME = "conditional_normalizing_flow"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the conditional normalizing flow on SMC posterior conditions."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_subfolder", choices=("unet", "unet_ema"), required=True)
    parser.add_argument("--eval_output_subdir", required=True)
    parser.add_argument("--primary_reference", required=True)
    parser.add_argument("--oracle_reference", required=True)
    add_continuous_dataset_arguments(parser)
    for n in (
        "eval_batch_size",
        "dataloader_num_workers",
        "num_posterior_samples",
        "num_visual_samples",
        "reference_repeats",
        "nearest_k",
        "w2_num_quantiles",
        "distance_chunk_size",
        "metric_seed",
        "source_seed",
    ):
        parser.add_argument(f"--{n}", type=int, required=True)
    parser.add_argument("--credible_interval_level", type=float, required=True)
    parser.add_argument("--kernel_bandwidth", type=float, default=None,
                        help="Fixed Gaussian bandwidth; default: independent simulator calibration median.")
    parser.add_argument("--reuse_existing", type=parse_bool, required=True)
    parser.add_argument("--peak_relative_threshold", type=float, required=True)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    bandwidth = getattr(args, "kernel_bandwidth", None)
    if bandwidth is not None and (not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("kernel_bandwidth must be finite and positive.")
    validate_continuous_dataset_arguments(args)
    for n in (
        "eval_batch_size",
        "num_posterior_samples",
        "num_visual_samples",
        "reference_repeats",
        "nearest_k",
        "w2_num_quantiles",
        "distance_chunk_size",
    ):
        if getattr(args, n) <= 0:
            raise ValueError(f"{n} must be positive.")
    if (
        args.dataloader_num_workers < 0
        or args.num_visual_samples > args.num_posterior_samples
        or args.num_posterior_samples <= args.nearest_k
    ):
        raise ValueError("Invalid evaluation counts.")
    for name in ("seed", "source_seed", "metric_seed"):
        if not 0 <= int(getattr(args, name)) < 2**63:
            raise ValueError(f"{name} must lie in [0,2**63).")
    if not 0.0 < args.credible_interval_level < 1.0:
        raise ValueError("credible_interval_level must lie in (0,1).")
    if not 0.0 <= args.peak_relative_threshold <= 1.0:
        raise ValueError("peak_relative_threshold must lie in [0,1].")
    if not all(
        math.isfinite(float(getattr(args, name)))
        for name in ("credible_interval_level", "peak_relative_threshold")
    ):
        raise ValueError("Floating-point evaluation controls must be finite.")
    relative_output = Path(args.eval_output_subdir)
    if (
        not args.output_dir.strip()
        or not args.checkpoint_path.strip()
        or not args.primary_reference.strip()
        or not args.oracle_reference.strip()
        or not args.eval_output_subdir.strip()
        or relative_output.is_absolute()
        or relative_output == Path(".")
        or ".." in relative_output.parts
    ):
        raise ValueError("Output/checkpoint/reference paths must be nonempty and the evaluation subdirectory must be safe and relative.")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    require_ema = args.model_subfolder == "unet_ema"
    if args.checkpoint_path.strip().lower() == "latest":
        return find_latest_checkpoint(Path(args.output_dir), require_ema=require_ema)
    requested = Path(args.checkpoint_path).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    return validate_checkpoint_path(
        requested, require_ema=require_ema, allow_legacy_numbered=False
    )


def _checkpoint_iteration(state: dict[str, Any]) -> int:
    value = state.get("global_iteration")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Checkpoint global_iteration must be nonnegative.")
    return int(value)


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
        )
    return nested_config_values_match(actual, expected)


def _validate_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _read_json_object(Path(args.output_dir) / "training_config.json")
    state = _read_json_object(checkpoint / "training_state.json")
    expected_values = {
        **continuous_dataset_config_values(args),
        "experiment": EXPERIMENT_NAME,
        "conditional_input_representation": CONDITIONAL_FLOW_INPUT_REPRESENTATION,
        "conditional_output_parameterization": CONDITIONAL_FLOW_OUTPUT_PARAMETERIZATION,
        "conditional_target_representation": CONDITIONAL_FLOW_TARGET_REPRESENTATION,
        "alr_reference_component": "last_spectrum_bin",
        "likelihood_measure": "Lebesgue_measure_in_fixed_ALR_chart",
    }
    saved = state.get("training_config")
    if not isinstance(root, dict) or not isinstance(saved, dict):
        raise ValueError("Training configuration artifacts must contain objects.")
    if state.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            f"Checkpoint experiment is {state.get('experiment')!r}, expected "
            f"{EXPERIMENT_NAME!r}."
        )
    for name, expected in expected_values.items():
        if name not in root or name not in saved:
            raise ValueError(f"Training configuration lacks {name!r}.")
        if not _matches(root[name], expected):
            raise ValueError(f"Root training configuration differs in {name!r}.")
        if not _matches(saved[name], expected):
            raise ValueError(f"Checkpoint training configuration differs in {name!r}.")
    _checkpoint_iteration(state)
    if root.get("network_architecture") != saved.get("network_architecture"):
        raise ValueError("Root and checkpoint CNF architecture contracts differ.")
    return root, state


def _architecture_from_saved_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct and validate the exact CNF architecture saved by training."""

    signature = config.get("network_architecture")
    if not isinstance(signature, dict):
        raise ValueError("Saved training configuration lacks CNF architecture.")
    required = (
        "num_spectrum_bins",
        "num_measurements",
        "num_coupling_layers",
        "hidden_dim",
        "context_dim",
        "num_condition_blocks",
        "scale_limit",
    )
    missing = [name for name in required if name not in signature]
    if missing:
        raise ValueError(f"Saved CNF architecture is missing {missing}.")
    architecture = {name: signature[name] for name in required}
    if signature != get_conditional_flow_architecture_signature(**architecture):
        raise ValueError("Saved CNF architecture signature is not canonical.")
    return architecture


def _safe_ratio(numerator: float, denominator: float, name: str) -> float:
    if denominator <= 0.0:
        raise RuntimeError(f"Cannot compute {name}: denominator is nonpositive.")
    return numerator / denominator


def _collective_raise_if_failed(
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
    method: str,
    training_config: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
    source_seed: int,
    deterministic: bool,
    inference_settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": SAMPLING_ARTIFACT_SCHEMA_VERSION,
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "method": method,
        "training_config": dict(training_config),
        "training_output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "weights": "ema" if args.model_subfolder == "unet_ema" else "raw",
        "model_subfolder": args.model_subfolder,
        "sampling_config": dict(sampling_config),
        "inference_settings": dict(inference_settings),
        "source_seed": int(source_seed),
        "num_samples_per_condition": int(args.num_posterior_samples),
        "state_size": int(args.num_spectrum_bins),
        "deterministic": bool(deterministic),
        "reference_path": str(Path(args.primary_reference).expanduser().resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = spec_from_args(args)
    checkpoint = _resolve_checkpoint(args)
    training_config, checkpoint_state = _validate_checkpoint(args, checkpoint)
    architecture = _architecture_from_saved_config(training_config)
    if architecture["num_spectrum_bins"] != args.num_spectrum_bins:
        raise ValueError("Saved CNF spectrum size differs from evaluation CLI.")
    if architecture["num_measurements"] != args.num_measurements:
        raise ValueError("Saved CNF measurement size differs from evaluation CLI.")
    sampling_config = {
        "flow_family": "conditional_realnvp_affine_coupling",
        "source_distribution": "standard_normal_alr",
        "num_posterior_samples": args.num_posterior_samples,
    }
    settings = EvaluationSettings(
        sample_count=args.num_posterior_samples,
        reference_repeats=args.reference_repeats,
        nearest_k=args.nearest_k,
        w2_num_quantiles=args.w2_num_quantiles,
        distance_chunk_size=args.distance_chunk_size,
        seed=args.metric_seed,
        credible_interval_level=args.credible_interval_level,
        kernel_bandwidth=getattr(args, "kernel_bandwidth", None),
    )
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed, device_specific=True)
    if not 0 <= int(args.source_seed) < 2**63 - accelerator.num_processes:
        raise ValueError("source seed plus process rank must lie in [0,2**63).")
    primary, oracle = load_and_validate_references(
        args.primary_reference, args.oracle_reference, spec
    )
    dataset = PosteriorConditionDataset(primary)
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
    provenance = _expected_cache_provenance(
        args=args,
        spec=spec,
        checkpoint=checkpoint,
        method=EXPERIMENT_NAME,
        training_config=training_config,
        sampling_config=sampling_config,
        source_seed=args.source_seed,
        deterministic=False,
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
    _collective_raise_if_failed(accelerator, reuse_error, "cached-sample validation")

    if artifact is None:
        model = load_conditional_flow_model_pretrained(
            checkpoint / args.model_subfolder
        )
        validate_conditional_flow_model_config(model, **architecture)
        model, loader = accelerator.prepare(
            model,
            DataLoader(
                dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.dataloader_num_workers,
                drop_last=False,
            ),
        )
        model.eval()
        evaluation_model = accelerator.unwrap_model(model)
        sampler = ConditionalNormalizingFlowSampler(
            evaluation_model,
            accelerator.device,
            source_seed=args.source_seed + accelerator.process_index,
        )
        gathered_indices: list[torch.Tensor] = []
        gathered_samples: list[torch.Tensor] = []
        diagnostic_totals: dict[str, float] = {}
        started = start_distributed_wall_timer(accelerator)
        for batch in loader:
            measurements = batch["measurement"]
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
        elapsed = slowest_distributed_elapsed_seconds(accelerator, started)
        post_error: Exception | None = None
        if accelerator.is_main_process:
            try:
                indices = torch.cat(gathered_indices)
                samples = torch.cat(gathered_samples)
                order = torch.argsort(indices)
                if not torch.equal(
                    indices[order], torch.sort(primary["condition_ids"].cpu()).values
                ):
                    raise RuntimeError(
                        "Distributed evaluation did not cover each condition once."
                    )
                sorted_samples = samples[order]
                sorted_ids = indices[order]
                positions = torch.searchsorted(
                    sorted_ids, primary["condition_ids"].cpu()
                )
                samples = sorted_samples[positions]
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
                    source_seed=args.source_seed,
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
                generated_count = diagnostic_totals["generated_state_count"]
                write_json(
                    evaluation_dir / "sampling_summary.json",
                    {
                        "experiment": EXPERIMENT_NAME,
                        "checkpoint_iteration": _checkpoint_iteration(checkpoint_state),
                        "conditional_flow_exact_inverse": True,
                        "actual_model_nfe_per_generated_sample": 1,
                        "inverse_coupling_layers_per_generated_sample": int(
                            architecture["num_coupling_layers"]
                        ),
                        "num_conditions": int(samples.shape[0]),
                        "num_posterior_samples": int(samples.shape[1]),
                        "inference_wall_seconds": elapsed,
                        "inference_seconds_per_generated_sample": elapsed
                        / generated_count,
                        "timing_interpretation": (
                            "maximum_rank_wall_time_divided_by_global_generated_"
                            "sample_count; throughput_equivalent_not_single_condition_latency"
                        ),
                        "sample_negative_fraction": _safe_ratio(
                            diagnostic_totals["negative_value_count"],
                            diagnostic_totals["generated_value_count"],
                            "sample_negative_fraction",
                        ),
                        "sample_nonfinite_fraction": _safe_ratio(
                            diagnostic_totals["nonfinite_value_count"],
                            diagnostic_totals["generated_value_count"],
                            "sample_nonfinite_fraction",
                        ),
                        "sample_mass_absolute_error_mean": _safe_ratio(
                            diagnostic_totals["mass_absolute_error_sum"],
                            generated_count,
                            "sample_mass_absolute_error_mean",
                        ),
                        "sample_alr_absolute_value_mean": _safe_ratio(
                            diagnostic_totals["alr_absolute_value_sum"],
                            diagnostic_totals["alr_value_count"],
                            "sample_alr_absolute_value_mean",
                        ),
                        "sampler_diagnostic_totals": diagnostic_totals,
                    },
                )
            except Exception as error:
                post_error = error
        _collective_raise_if_failed(accelerator, post_error, "sample postprocessing")

    metric_error: Exception | None = None
    if accelerator.is_main_process:
        try:
            assert artifact is not None
            evaluate_and_write_metrics(
                artifact,
                primary,
                oracle,
                evaluation_dir,
                settings=settings,
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
                    "Conditional normalizing flow | "
                    f"K={args.num_posterior_samples}"
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
    _collective_raise_if_failed(
        accelerator, metric_error, "metric and figure postprocessing"
    )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
