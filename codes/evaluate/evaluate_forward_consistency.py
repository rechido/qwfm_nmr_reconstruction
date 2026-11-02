"""Evaluate individual-draw and mean-spectrum errors from saved predictions.

CPU only. No model weights, neural inference, test simulation, SMC, or ES/KS
calculation is needed. All stored draws are used, including K=1 for regression.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from codes.data.grid import build_nmr_grid  # noqa: E402
from codes.data.spec import ContinuousNMRSpec  # noqa: E402
from codes.metrics.forward_consistency import (  # noqa: E402
    FORWARD_CONSISTENCY_PROTOCOL, METRICS, MEAN_SPECTRUM_METRICS,
    MEAN_SPECTRUM_PROTOCOL, mean_spectrum_forward_errors,
    per_sample_forward_errors, summarize_forward_errors,
)
from codes.utils.io_utils import atomic_torch_save, atomic_write_json  # noqa: E402
from codes.utils.model_variant_utils import FLOW_METHODS, VARIANTS, experiment_dimensions  # noqa: E402
from codes.utils.predictive_generation_utils import (  # noqa: E402
    PROPER_SCORE_PROTOCOL_VERSION, PROPER_SCORE_TEST_SPLIT, _verified_cache, file_sha256,
)
from codes.utils.proper_score_locks import file_lock  # noqa: E402


LABELS = {
    "histogram_linear": "HFM", "quantile_wasserstein_l128": "QWFM L128",
    "quantile_wasserstein_l256": "QWFM L256", "quantile_wasserstein_l512": "QWFM L512",
    "deterministic_regression": "Regression MSE",
    "deterministic_regression_mse_w1": "Regression MSE+W1",
    "conditional_vae_z16": "CVAE z16", "conditional_vae_z32": "CVAE z32",
    "conditional_normalizing_flow": "CNF",
}
FORMULAS = {
    "noisy_mse": "mean_m (sum_n A[m,n]*h_hat[a,k,n] - y[a,m])^2",
    "clean_mse": "mean_m (sum_n A[m,n]*(h_hat[a,k,n]-h[a,n]))^2",
    "noisy_normalized_mse": "noisy_mse / sigma^2",
    "clean_normalized_mse": "clean_mse / sigma^2",
}
FORMULAS.update({"mean_spectrum_" + name: "Replace h_hat[a,k] by mean_k h_hat[a,k]: " + formula
                 for name, formula in tuple(FORMULAS.items())})


def _csv_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(","))
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of unique values.")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment_dir", type=Path)
    parser.add_argument("--run_dir", type=Path,
                        help="Existing proper_scores/g16384_k8 directory containing saved truths and draws.")
    parser.add_argument("--output_dir", type=Path,
                        help="Defaults to <experiment>/metric_comparisons/forward_consistency/<run name>.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--variant", choices=tuple(VARIANTS), help="One model to evaluate from saved samples.")
    selection.add_argument("--variants", type=_csv_list, help=argparse.SUPPRESS)
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--nfe", nargs="+", type=int,
                       help="Restrict scoring to saved FM samples at these NFE values.")
    steps.add_argument("--eval_ode_steps", type=_csv_list, help=argparse.SUPPRESS)
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--sample_count", type=int, default=8,
                        help="Required stored stochastic K; all draws are used, regression must store one.")
    parser.add_argument("--condition_batch_size", type=int, default=256)
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute reports, including when saved source files have changed.")
    args = parser.parse_args(argv)
    for name in ("num_conditions", "sample_count", "condition_batch_size", "num_cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    args.variants = (args.variant,) if args.variant is not None else args.variants
    if any(variant not in VARIANTS for variant in args.variants):
        parser.error(f"Unknown variant. Supported variants: {', '.join(VARIANTS)}")
    try:
        values = args.nfe if args.nfe is not None else args.eval_ode_steps
        args.eval_ode_steps = None if values is None else tuple(int(value) for value in values)
        if args.eval_ode_steps is not None:
            if min(args.eval_ode_steps) < 1 or len(set(args.eval_ode_steps)) != len(args.eval_ode_steps):
                raise ValueError
    except ValueError:
        parser.error("--nfe must contain unique positive integers.")
    if args.experiment_dir is None and args.run_dir is None:
        parser.error("Specify --experiment_dir or --run_dir.")
    if args.experiment_dir is None:
        args.experiment_dir = args.run_dir.expanduser().resolve().parent.parent
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    name = f"g{args.num_conditions}_k{args.sample_count}"
    args.run_dir = (args.run_dir or args.experiment_dir / "proper_scores" / name).expanduser().resolve()
    try:
        args.dimensions = experiment_dimensions(args.experiment_dir)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir = (args.output_dir or args.experiment_dir / "metric_comparisons" / "forward_consistency" / args.run_dir.name).expanduser().resolve()
    if args.output_dir == args.run_dir or args.run_dir in args.output_dir.parents:
        parser.error("Reports must be outside the saved-input run directory.")
    return args


def _equal_tensor(first: object, second: torch.Tensor, name: str) -> None:
    if not isinstance(first, torch.Tensor) or first.shape != second.shape or not torch.equal(first, second):
        raise ValueError(f"{name} differs from the saved test conditions.")


def _load_verified_input(path: Path) -> tuple[dict, str]:
    before = path.stat()
    artifact = _verified_cache(path, True)
    digest = file_sha256(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ValueError(f"Saved input changed while it was being read: {path}; finish its writer and retry.")
    return artifact, digest


def _validate_grid(grid: dict, expected: dict) -> None:
    if not isinstance(grid, dict) or grid.get("specification_hash") != expected["specification_hash"]:
        raise ValueError("Grid specification hash differs.")
    for name, value in expected.items():
        if not isinstance(value, torch.Tensor):
            continue
        actual = grid.get(name)
        if not isinstance(actual, torch.Tensor) or actual.shape != value.shape:
            raise ValueError(f"Missing or incompatible grid field: {name}.")
        if not torch.allclose(actual.double(), value.double(), rtol=0, atol=1e-12):
            raise ValueError(f"Grid field {name} differs from the observation model.")


def validate_truth(truth: dict, groups: int) -> tuple[ContinuousNMRSpec, dict]:
    spec = ContinuousNMRSpec.from_dict(truth["specification"])
    provenance = truth.get("proper_score_provenance", {})
    if (truth.get("protocol_id") != spec.protocol_id
            or truth.get("specification_hash") != spec.specification_hash
            or truth.get("split") != PROPER_SCORE_TEST_SPLIT
            or provenance.get("protocol_version") != PROPER_SCORE_PROTOCOL_VERSION
            or provenance.get("split") != PROPER_SCORE_TEST_SPLIT
            or provenance.get("num_conditions") != groups
            or provenance.get("master_seed") != truth.get("master_seed")
            or provenance.get("specification_hash") != spec.specification_hash):
        raise ValueError("Expected the original independent proper-score test conditions.")
    _equal_tensor(truth.get("sample_id"), torch.arange(groups), "Truth condition IDs")
    _validate_grid(truth.get("grid"), build_nmr_grid(spec).as_artifact())
    for name, width in (("histogram", spec.num_spectrum_bins),
                        ("measurement", spec.num_measurements),
                        ("clean_measurement", spec.num_measurements),
                        ("noise", spec.num_measurements)):
        value = truth.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != (groups, width)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid or missing saved truth field: {name}.")
    clean = truth["histogram"].double() @ truth["grid"]["kernel"].double().T
    clean_difference = float((clean - truth["clean_measurement"].double()).abs().max())
    observation_difference = float((truth["clean_measurement"].double() + truth["noise"].double()
                                    - truth["measurement"].double()).abs().max())
    # Input signal arrays were stored in float32 by the simulator.
    if clean_difference > 2e-6 or observation_difference > 2e-6:
        raise ValueError("Saved test pairs do not satisfy clean=A h and y=clean+noise.")
    return spec, {
        "max_abs_saved_clean_minus_Ah": clean_difference,
        "max_abs_saved_observation_minus_clean_plus_noise": observation_difference,
        "float32_input_consistency_atol": 2e-6,
    }


def validate_samples(artifact: dict, truth: dict, truth_hash: str,
                     variant: str, nfe: int | None, budget: int) -> dict:
    provenance = artifact.get("proper_score_provenance", {})
    spec = ContinuousNMRSpec.from_dict(truth["specification"])
    method = VARIANTS[variant][0]
    deterministic = method.startswith("deterministic_regression")
    used = 1 if deterministic else budget
    if (artifact.get("protocol_id") != spec.protocol_id
            or artifact.get("specification_hash") != spec.specification_hash
            or ContinuousNMRSpec.from_dict(artifact["specification"]).specification_hash != spec.specification_hash
            or artifact.get("method") != method
            or artifact.get("deterministic") is not deterministic
            or provenance.get("protocol_version") != PROPER_SCORE_PROTOCOL_VERSION
            or provenance.get("variant") != variant
            or provenance.get("method") != method
            or provenance.get("conditions_sha256") != truth_hash
            or provenance.get("requested_sample_count") != budget
            or provenance.get("stored_sample_count") != used):
        raise ValueError("Predictions have incompatible model, test conditions, or sample count.")
    if method in FLOW_METHODS and (artifact.get("sampling_config", {}).get("ode_steps") != nfe
                                  or provenance.get("nfe") != nfe):
        raise ValueError("Prediction NFE differs from the requested existing sampling setting.")
    _equal_tensor(artifact.get("condition_ids"), truth["sample_id"], "Prediction condition IDs")
    _equal_tensor(artifact.get("measurements"), truth["measurement"], "Prediction observations")
    _validate_grid(artifact.get("grid"), truth["grid"])
    expected_shape = (len(truth["sample_id"]), used, spec.num_spectrum_bins)
    if not isinstance(artifact.get("samples"), torch.Tensor) or artifact["samples"].shape != expected_shape:
        raise ValueError(f"Predictions must have shape {expected_shape}.")
    return provenance


def _write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    # Writers hold the shared comparison directory's publication lock.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8")
    temporary.replace(path)


def _condition_rows(ids: torch.Tensor, errors: dict[str, torch.Tensor]) -> list[dict]:
    means = {name: values.mean(1).tolist() for name, values in errors.items()}
    return [{"condition_id": int(condition_id), **{name: values[index] for name, values in means.items()}}
            for index, condition_id in enumerate(ids)]


def _summary_row(report: dict) -> dict:
    row = {key: report[key] for key in ("variant", "method_label", "nfe", "condition_count", "sample_count")}
    for name in (*METRICS, *MEAN_SPECTRUM_METRICS):
        stats = report["aggregate"][name]
        row[name] = stats["mean"]
        row[name + "_condition_se"] = stats["condition_standard_error"]
        row[name + "_sample_q95"] = stats["pooled_sample_q95"]
    return row


def _validate_report_outputs(report: dict, output: Path) -> None:
    names = {"per_sample_errors.pt", "per_condition.csv"}
    if not all((output / name).is_file() for name in names):
        raise ValueError(f"Incomplete output at {output}; rerun with --recompute.")
    checksums = report.get("output_sha256")
    if not isinstance(checksums, dict) or set(checksums) != names:
        raise ValueError(f"Missing output checksums at {output}; rerun with --recompute.")
    for name, digest in checksums.items():
        if file_sha256(output / name) != digest:
            raise ValueError(f"Changed output {output / name}; rerun with --recompute.")


def _score(path: Path, variant: str, nfe: int | None, truth: dict, common: dict,
           output: Path, args: argparse.Namespace) -> None:
    artifact, source_hash = _load_verified_input(path)
    provenance = validate_samples(artifact, truth, common["truth_sha256"], variant, nfe, args.sample_count)
    identity = {**common, "variant": variant, "method_label": LABELS[variant], "nfe": nfe,
                "sample_count": artifact["samples"].shape[1], "samples_path": str(path),
                "samples_sha256": source_hash, "generation_provenance": provenance}
    report_path = output / "forward_consistency.json"
    if report_path.is_file() and not args.recompute:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if any(report.get(name) != value for name, value in identity.items()):
            raise ValueError(f"Source/configuration changed for {report_path}; use --recompute or a new output directory.")
        _validate_report_outputs(report, output)
        print("  reused checked forward-error report", flush=True)
        return

    errors = per_sample_forward_errors(
        artifact["samples"], truth["histogram"], truth["measurement"], truth["grid"]["kernel"],
        noise_standard_deviation=common["noise_standard_deviation"], condition_batch_size=args.condition_batch_size,
    )
    errors.update(mean_spectrum_forward_errors(
        artifact["samples"], truth["histogram"], truth["measurement"], truth["grid"]["kernel"],
        noise_standard_deviation=common["noise_standard_deviation"], condition_batch_size=args.condition_batch_size,
    ))
    report = {**identity, "aggregate": summarize_forward_errors(errors)}
    condition_rows = _condition_rows(truth["sample_id"], errors)
    # Different models calculate independently. Publish the related files
    # together so comparison readers never see a partially replaced report.
    with file_lock(args.output_dir / ".publication.lock", blocking=True):
        atomic_torch_save(output / "per_sample_errors.pt", {
            "protocol_version": FORWARD_CONSISTENCY_PROTOCOL,
            "condition_ids": truth["sample_id"], "truth_sha256": common["truth_sha256"],
            "samples_sha256": source_hash, "noise_standard_deviation": common["noise_standard_deviation"],
            "errors": errors,
        })
        _write_csv(output / "per_condition.csv", condition_rows)
        report["output_sha256"] = {name: file_sha256(output / name)
                                   for name in ("per_sample_errors.pt", "per_condition.csv")}
        atomic_write_json(report_path, report)
    print(f"  mean normalized MSE: noisy={report['aggregate']['noisy_normalized_mse']['mean']:.6f}, "
          f"clean={report['aggregate']['clean_normalized_mse']['mean']:.6f}", flush=True)


def _collect_reports(output: Path, common: dict) -> list[dict]:
    """Retain all completed models/NFE settings, independent of this invocation."""
    records = []
    for path in sorted((output / "models").rglob("forward_consistency.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if any(report.get(name) != value for name, value in common.items()):
            raise ValueError(f"Incompatible reports must not be pooled: {path}; use a new output directory.")
        source = Path(report["samples_path"])
        if not source.is_file() or file_sha256(source) != report["samples_sha256"]:
            raise ValueError(f"Prediction source changed after scoring: {source}; reevaluate that model with --recompute.")
        _validate_report_outputs(report, path.parent)
        records.append({**report, "report_path": str(path)})
    order = {variant: index for index, variant in enumerate(VARIANTS)}
    records.sort(key=lambda report: (order[report["variant"]], report["nfe"] or 0))
    return records


def _run(args: argparse.Namespace) -> dict:
    truth_path = args.run_dir / "test_conditions.pt"
    if not truth_path.is_file():
        raise FileNotFoundError(f"Missing saved test conditions: {truth_path}; this evaluator never creates them.")
    truth, truth_hash = _load_verified_input(truth_path)
    spec, checks = validate_truth(truth, args.num_conditions)
    if (spec.num_measurements, spec.num_spectrum_bins) != args.dimensions:
        raise ValueError("Run directory dimensions do not match the saved simulator specification.")
    shape = f"m{spec.num_measurements}_n{spec.num_spectrum_bins}"
    jobs = []
    for variant in args.variants:
        method, pattern, _, _ = VARIANTS[variant]
        model_name = pattern.format(shape=shape)
        if method in FLOW_METHODS:
            steps = args.eval_ode_steps
            if steps is None:
                steps = tuple(sorted(int(path.parent.name.removeprefix("nfe_"))
                                     for path in (args.run_dir / model_name).glob("nfe_[0-9]*/samples.pt")))
                if not steps:
                    raise FileNotFoundError(f"Missing saved predictions for {variant}. No inference is run automatically.")
        else:
            steps = (None,)
        for nfe in steps:
            relative = Path(model_name)
            if nfe is not None:
                relative /= f"nfe_{nfe:04d}"
            path = args.run_dir / relative / "samples.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved predictions: {path}. No inference is run automatically.")
            jobs.append((variant, nfe, relative, path))

    common = {
        "protocol_version": FORWARD_CONSISTENCY_PROTOCOL,
        "mean_spectrum_protocol_version": MEAN_SPECTRUM_PROTOCOL,
        "condition_count": args.num_conditions,
        "requested_sample_count": args.sample_count,
        "num_measurements": spec.num_measurements,
        "num_spectrum_bins": spec.num_spectrum_bins,
        "noise_standard_deviation": spec.noise_standard_deviation,
        "specification_hash": spec.specification_hash,
        "truth_path": str(truth_path), "truth_sha256": truth_hash,
        "truth_consistency_checks": checks,
        "formulas": FORMULAS,
        "aggregation": "Individual: score draws then average; mean spectrum: average draws then score; equal weight per condition",
        "uncertainty_unit": "independent test condition (not individual generated draw)",
        "ci_method": "approximate normal 95% interval: condition mean +/- 1.96*condition SE",
        "root_mean_mse_definition": "sqrt(overall mean MSE), not mean of per-draw RMSE",
        "prediction_noise_added": False, "network_forward_performed": False,
        "test_data_generated": False, "smc_posterior_samples_used": False,
        "es_ks_computed": False, "k_sweep_performed": False,
        "notes": [
            "Noisy error measures observation fit; smaller is not necessarily a better posterior.",
            "Clean error equals comparing y with A h_hat plus the original noise, up to input rounding.",
            "The clean reference is A h recomputed from the saved kernel and truth in float64.",
            "Every saved draw is evaluated; no best-of selection or mass repair.",
            "Mean-spectrum errors separately score the arithmetic mean of all draws; they do not replace individual-draw errors.",
        ],
    }
    summary_path = args.output_dir / "forward_consistency_summary.json"
    if summary_path.is_file() and not args.recompute:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(previous.get(name) != value for name, value in common.items()):
            raise ValueError("Test conditions/evaluation configuration changed; use --recompute or a new output directory.")
    baseline_errors = per_sample_forward_errors(
        truth["histogram"][:, None, :], truth["histogram"], truth["measurement"], truth["grid"]["kernel"],
        noise_standard_deviation=spec.noise_standard_deviation, condition_batch_size=args.condition_batch_size,
    )
    baseline_errors.update(mean_spectrum_forward_errors(
        truth["histogram"][:, None, :], truth["histogram"], truth["measurement"], truth["grid"]["kernel"],
        noise_standard_deviation=spec.noise_standard_deviation, condition_batch_size=args.condition_batch_size,
    ))
    baseline = {**common, "variant": "ground_truth", "method_label": "Ground truth",
                "nfe": None, "sample_count": 1, "aggregate": summarize_forward_errors(baseline_errors)}
    for number, (variant, nfe, relative, path) in enumerate(jobs, 1):
        print(f"[{number}/{len(jobs)}] forward errors only: {relative}", flush=True)
        output = args.output_dir / "models" / relative
        with file_lock(output / ".score.lock", blocking=True):
            _score(path, variant, nfe, truth, common, output, args)

    baseline_rows = _condition_rows(truth["sample_id"], baseline_errors)
    # Re-read every completed report under the publication lock. A later
    # per-model or NFE-filtered call must not replace earlier comparison rows.
    with file_lock(args.output_dir / ".publication.lock", blocking=True):
        records = _collect_reports(args.output_dir, common)
        summary = {
            **common,
            "variants": list(dict.fromkeys(report["variant"] for report in records)),
            "eval_ode_steps": sorted({report["nfe"] for report in records if report["nfe"] is not None}),
            "ground_truth": baseline, "records": records, "model_setting_count": len(records),
            "comparison_scope": "All compatible completed forward-error reports in this output directory; missing models are not inferred.",
        }
        atomic_write_json(args.output_dir / "ground_truth.json", baseline)
        _write_csv(args.output_dir / "ground_truth_per_condition.csv", baseline_rows)
        atomic_write_json(summary_path, summary)
        _write_csv(args.output_dir / "forward_consistency_summary.csv", [_summary_row(baseline), *map(_summary_row, records)])
    print(f"Saved: {args.output_dir / 'forward_consistency_summary.csv'}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(args.num_cpu_threads)
    started = time.perf_counter()
    summary = _run(args)
    print(f"Finished offline evaluation in {time.perf_counter() - started:.1f}s (CPU; no new inference).", flush=True)
    return summary


if __name__ == "__main__":
    main()
