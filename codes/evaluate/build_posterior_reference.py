# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.data.reference_smc import SMCSettings, run_reference_for_condition
from codes.data.simulator import generate_fixed_split
from codes.utils.argument_utils import add_spec_arguments, spec_from_args
from codes.utils.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    load_torch_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build independent tempered-SMC reference posterior banks."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    add_spec_arguments(parser)
    parser.add_argument("--num_test_conditions", type=int, default=32)
    parser.add_argument("--test_dataset_seed", type=int, default=2026090301)
    parser.add_argument("--reference_master_seed", type=int, default=2026090302)
    parser.add_argument("--run_ids", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--condition_start", type=int, default=0)
    parser.add_argument("--condition_stop", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    parser.add_argument("--num_particles", type=int, default=4096)
    parser.add_argument("--output_samples", type=int, default=4096)
    parser.add_argument("--target_conditional_ess_fraction", type=float, default=0.95)
    parser.add_argument("--resample_ess_fraction", type=float, default=0.5)
    parser.add_argument("--rejuvenation_sweeps", type=int, default=25)
    parser.add_argument("--tempering_adaptation_fraction", type=float, default=0.5)
    parser.add_argument("--final_rejuvenation_sweeps", type=int, default=10)
    parser.add_argument("--maximum_final_rejuvenation_sweeps", type=int, default=128)
    parser.add_argument("--final_diagnostic_interval", type=int, default=5)
    parser.add_argument("--final_acceptance_window", type=int, default=5)
    parser.add_argument("--final_minimum_mean_acceptance", type=float, default=0.10)
    parser.add_argument("--final_minimum_unique_fraction", type=float, default=0.999)
    parser.add_argument(
        "--final_maximum_k3_zero_radius_fraction", type=float, default=0.0
    )
    parser.add_argument("--maximum_temperatures", type=int, default=512)
    parser.add_argument("--beta_bisection_iterations", type=int, default=48)
    parser.add_argument("--center_logit_step", type=float, default=0.15)
    parser.add_argument("--log_cv_step_fraction", type=float, default=0.05)
    parser.add_argument("--weight_logit_step", type=float, default=0.25)
    parser.add_argument("--adaptation_target_acceptance", type=float, default=0.30)
    parser.add_argument("--adaptation_gain", type=float, default=0.50)
    parser.add_argument("--minimum_step_scale", type=float, default=1.0e-6)
    parser.add_argument("--maximum_step_scale", type=float, default=2.0)
    parser.add_argument("--overwrite_conditions", action="store_true")
    args = parser.parse_args(argv)
    if args.num_test_conditions <= 0:
        parser.error("--num_test_conditions must be positive.")
    if args.num_cpu_threads <= 0:
        parser.error("--num_cpu_threads must be positive.")
    if args.condition_start < 0:
        parser.error("--condition_start must be nonnegative.")
    stop = (
        args.num_test_conditions if args.condition_stop is None else args.condition_stop
    )
    if not args.condition_start < stop <= args.num_test_conditions:
        parser.error("Require condition_start < condition_stop <= num_test_conditions.")
    if len(set(args.run_ids)) != len(args.run_ids) or any(
        value < 0 for value in args.run_ids
    ):
        parser.error("--run_ids must contain unique nonnegative integers.")
    return args


def _settings_from_args(args: argparse.Namespace) -> SMCSettings:
    return SMCSettings(
        num_particles=args.num_particles,
        target_conditional_ess_fraction=args.target_conditional_ess_fraction,
        resample_ess_fraction=args.resample_ess_fraction,
        rejuvenation_sweeps=args.rejuvenation_sweeps,
        tempering_adaptation_fraction=args.tempering_adaptation_fraction,
        final_rejuvenation_sweeps=args.final_rejuvenation_sweeps,
        maximum_final_rejuvenation_sweeps=(args.maximum_final_rejuvenation_sweeps),
        final_diagnostic_interval=args.final_diagnostic_interval,
        final_acceptance_window=args.final_acceptance_window,
        final_minimum_mean_acceptance=args.final_minimum_mean_acceptance,
        final_minimum_unique_fraction=args.final_minimum_unique_fraction,
        final_maximum_k3_zero_radius_fraction=(
            args.final_maximum_k3_zero_radius_fraction
        ),
        maximum_temperatures=args.maximum_temperatures,
        beta_bisection_iterations=args.beta_bisection_iterations,
        center_logit_step=args.center_logit_step,
        log_cv_step_fraction=args.log_cv_step_fraction,
        weight_logit_step=args.weight_logit_step,
        adaptation_target_acceptance=args.adaptation_target_acceptance,
        adaptation_gain=args.adaptation_gain,
        minimum_step_scale=args.minimum_step_scale,
        maximum_step_scale=args.maximum_step_scale,
        output_samples=args.output_samples,
    )


def _validate_condition_cache(
    payload: dict[str, Any],
    *,
    specification_hash: str,
    settings: SMCSettings,
    run_id: int,
    condition_index: int,
    condition_id: int,
) -> None:
    expected = {
        "specification_hash": specification_hash,
        "smc_settings": settings.to_dict(),
        "reference_run_id": run_id,
        "condition_index": condition_index,
        "condition_id": condition_id,
    }
    mismatches = [
        f"{key}: cached={payload.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Incompatible cached condition artifact:\n" + "\n".join(mismatches)
        )


def _condition_payload(
    result: Any,
    *,
    spec: Any,
    settings: SMCSettings,
    run_id: int,
    condition_index: int,
    condition_id: int,
) -> dict[str, Any]:
    payload = {
        "protocol_id": spec.protocol_id,
        "specification_hash": spec.specification_hash,
        "smc_settings": settings.to_dict(),
        "reference_run_id": run_id,
        "condition_index": condition_index,
        "condition_id": condition_id,
        "reference_histograms": result.reference_histograms,
        "reference_component_counts": result.reference_component_counts,
        "reference_active_mask": result.reference_active_mask,
        "reference_centers": result.reference_centers,
        "reference_coefficients_of_variation": result.reference_coefficients_of_variation,
        "reference_mixture_weights": result.reference_mixture_weights,
        "posterior_component_count_probability": result.posterior_component_count_probability,
        "log_evidence": result.log_evidence,
        "diagnostics": result.diagnostics,
    }
    return payload


def _compile_run(
    run_directory: Path,
    *,
    test_split: dict[str, Any],
    spec: Any,
    settings: SMCSettings,
    run_id: int,
    reference_master_seed: int,
) -> Path | None:
    condition_payloads: list[dict[str, Any]] = []
    for index in range(int(test_split["measurement"].shape[0])):
        path = run_directory / f"condition_{index:04d}.pt"
        if not path.is_file():
            return None
        payload = load_torch_artifact(path)
        _validate_condition_cache(
            payload,
            specification_hash=spec.specification_hash,
            settings=settings,
            run_id=run_id,
            condition_index=index,
            condition_id=int(test_split["sample_id"][index]),
        )
        condition_payloads.append(payload)

    def stack(name: str) -> torch.Tensor:
        return torch.stack([payload[name] for payload in condition_payloads], dim=0)

    artifact = {
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "smc_settings": settings.to_dict(),
        "reference_run_id": run_id,
        "reference_master_seed": reference_master_seed,
        "condition_ids": test_split["sample_id"],
        "measurements": test_split["measurement"],
        "reference_histograms": stack("reference_histograms"),
        "reference_component_counts": stack("reference_component_counts"),
        "reference_active_mask": stack("reference_active_mask"),
        "reference_centers": stack("reference_centers"),
        "reference_coefficients_of_variation": stack(
            "reference_coefficients_of_variation"
        ),
        "reference_mixture_weights": stack("reference_mixture_weights"),
        "posterior_component_count_probability": stack(
            "posterior_component_count_probability"
        ),
        "log_evidence": stack("log_evidence"),
        "reference_duplicate_diagnostics": [
            payload["diagnostics"]["compiled_reference_duplicate_diagnostics"]
            for payload in condition_payloads
        ],
        "grid": test_split["grid"],
    }
    artifact["truth_histograms"] = test_split["histogram"]
    artifact["truth_component_counts"] = test_split["component_count"]
    destination = run_directory / f"reference_run_{run_id:02d}.pt"
    atomic_torch_save(destination, artifact)
    atomic_write_json(
        run_directory / f"reference_run_{run_id:02d}_diagnostics.json",
        [payload["diagnostics"] for payload in condition_payloads],
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.num_cpu_threads)
    spec = spec_from_args(args)
    settings = _settings_from_args(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "num_test_conditions": args.num_test_conditions,
        "test_dataset_seed": args.test_dataset_seed,
        "reference_master_seed": args.reference_master_seed,
        "smc_settings": settings.to_dict(),
    }
    config_path = output / "reference_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != config:
            raise ValueError(
                f"Existing reference configuration differs at {config_path}. "
                "Use a new output directory."
            )
    else:
        atomic_write_json(config_path, config)

    test_path = output / "test_conditions.pt"
    if test_path.exists():
        test_split = load_torch_artifact(test_path)
        if test_split.get("specification_hash") != spec.specification_hash:
            raise ValueError("Cached test conditions use a different specification.")
        if int(test_split["measurement"].shape[0]) != args.num_test_conditions:
            raise ValueError("Cached test condition count differs from the request.")
    else:
        test_split = generate_fixed_split(
            num_samples=args.num_test_conditions,
            spec=spec,
            master_seed=args.test_dataset_seed,
            split="posterior_test",
        )
        atomic_torch_save(test_path, test_split)

    stop = (
        args.num_test_conditions if args.condition_stop is None else args.condition_stop
    )
    for run_id in args.run_ids:
        run_directory = output / f"run_{run_id:02d}"
        run_directory.mkdir(parents=True, exist_ok=True)
        for condition_index in range(args.condition_start, stop):
            condition_id = int(test_split["sample_id"][condition_index])
            destination = run_directory / f"condition_{condition_index:04d}.pt"
            if destination.exists() and not args.overwrite_conditions:
                cached = load_torch_artifact(destination)
                _validate_condition_cache(
                    cached,
                    specification_hash=spec.specification_hash,
                    settings=settings,
                    run_id=run_id,
                    condition_index=condition_index,
                    condition_id=condition_id,
                )
                print(
                    f"reuse run={run_id} condition={condition_index}: {destination}",
                    flush=True,
                )
                continue
            print(
                f"start run={run_id} condition={condition_index}/{args.num_test_conditions - 1}",
                flush=True,
            )
            started = time.perf_counter()
            result = run_reference_for_condition(
                test_split["measurement"][condition_index],
                condition_id=condition_id,
                spec=spec,
                settings=settings,
                master_seed=args.reference_master_seed,
                run_id=run_id,
                device=args.device,
            )
            atomic_torch_save(
                destination,
                _condition_payload(
                    result,
                    spec=spec,
                    settings=settings,
                    run_id=run_id,
                    condition_index=condition_index,
                    condition_id=condition_id,
                ),
            )
            elapsed = time.perf_counter() - started
            print(f"saved {destination} ({elapsed:.1f} s)", flush=True)
        compiled = _compile_run(
            run_directory,
            test_split=test_split,
            spec=spec,
            settings=settings,
            run_id=run_id,
            reference_master_seed=args.reference_master_seed,
        )
        if compiled is None:
            print(
                f"run={run_id} is a partial shard; compile after all conditions exist.",
                flush=True,
            )
        else:
            print(f"compiled {compiled}", flush=True)


if __name__ == "__main__":
    main()
