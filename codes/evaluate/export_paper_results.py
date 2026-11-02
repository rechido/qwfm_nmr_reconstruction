"""Export manuscript figures and tables from completed evaluation reports.

Reads JSON/CSV reports only. Never loads spectra, checkpoints, or reference
tensors, and never runs inference, simulation, or forward-error evaluation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


SETTINGS = ((32, 128), (64, 128), (128, 256))
MODELS = ("HFM", "QWFM", "Regression", "CVAE")
SCORE_PROTOCOLS = {
    ("energy_score", "l2"): "truth_histogram_l2_energy_score_u_v1",
    ("energy_score", "w2"): "truth_histogram_w2_exact_energy_score_u_v1",
    ("kernel_score", "l2"): "truth_histogram_gaussian_kernel_score_u_v1",
    ("kernel_score", "w2"): "truth_histogram_w2_exact_gaussian_kernel_score_u_v1",
}


@dataclass(frozen=True)
class PaperProtocol:
    settings: tuple[tuple[int, int], ...] = SETTINGS
    nfes: tuple[int, ...] = (1, 4, 16, 64, 256)
    pr_conditions: int = 32
    reference_samples: int = 256
    reference_repeats: int = 20
    nearest_k: int = 3
    pr_w2_quantiles: int = 2048
    score_conditions: int = 16384
    score_samples: int = 8
    calibration_count: int = 512
    checkpoint_iteration: int = 200000


def model_identity(m: int, n: int, model: str) -> tuple[str, str]:
    shape = f"m{m}_n{n}"
    return {
        "HFM": ("histogram_linear", f"histogram_linear_{shape}_unet64x4_t1024"),
        "QWFM": (f"quantile_wasserstein_l{n}", f"quantile_wasserstein_{shape}_l{n}_unet64x4_t1024"),
        "Regression": ("deterministic_regression_mse_w1", f"deterministic_regression_mse_w1_{shape}_unet64x4"),
        "CVAE": ("conditional_vae_z16", f"conditional_vae_{shape}_unet64x4_z16"),
    }[model]


def _require(actual: object, expected: object, description: str, path: Path) -> None:
    if actual != expected:
        raise ValueError(f"{path}: incompatible {description}: expected {expected!r}, found {actual!r}.")


def _nonempty(value: object, description: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: missing {description}.")
    return value


def _consistent(bank: dict, key: object, value: object, description: str, path: Path) -> None:
    if key in bank:
        _require(value, bank[key], description, path)
    else:
        bank[key] = value


class Reports:
    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def read_text(self, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation report: {path}. Run the corresponding offline evaluation first.")
        content = path.read_bytes()
        self.hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8")

    def json(self, path: Path) -> dict:
        result = json.loads(self.read_text(path))
        if not isinstance(result, dict):
            raise ValueError(f"{path}: expected a JSON report object.")
        return result


def _statistics(values: list[float], reported: dict, path: Path) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path}: missing or nonfinite per-condition metric values.")
    mean = math.fsum(values) / len(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    if not math.isclose(mean, reported.get("mean", math.nan), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{path}: saved aggregate differs from per-condition values.")
    for key in ("standard_error", "condition_standard_error"):
        if key in reported and reported[key] is not None:
            if not math.isclose(se, reported[key], rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError(f"{path}: saved {key} differs from condition-based uncertainty.")
    return mean, se


def _generation(report: dict, model: str, variant: str, nfe: int | None,
                path: Path, protocol: PaperProtocol) -> dict:
    generation = report.get("generation_provenance", {})
    _require(generation.get("protocol_version"), "proper_score_generation_v1", "generation protocol", path)
    _require(generation.get("variant"), variant, "generation variant", path)
    # The shared generator records one call for non-flow methods; reports use
    # null NFE because their predictions do not depend on an ODE step budget.
    _require(generation.get("nfe"), nfe if nfe is not None else 1, "generation NFE", path)
    _require(generation.get("model_subfolder"), "unet_ema", "EMA model", path)
    checkpoint = generation.get("checkpoint_identity", {})
    _nonempty(checkpoint.get("sha256"), "checkpoint identity", path)
    if "model_subfolder" in checkpoint:
        _require(checkpoint["model_subfolder"], "unet_ema", "checkpoint EMA model", path)
    for record in (report, generation, checkpoint):
        if "checkpoint_iteration" in record:
            _require(record["checkpoint_iteration"], protocol.checkpoint_iteration, "checkpoint iteration", path)
    if model == "QWFM":
        _require(generation.get("decoder_signature", {}).get("decoder"), "linear_cdf", "QWFM decoder", path)
    return generation


def collect_reports(experiments_root: Path, protocol: PaperProtocol = PaperProtocol(), *,
                    models: tuple[str, ...] = MODELS) -> tuple[list[dict], dict]:
    """Validate and aggregate only the explicitly selected models/settings."""
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError(f"Select unique models from {MODELS}.")
    reports = Reports()
    rows: list[dict] = []
    identities: dict = {}
    unavailable_metadata: list[str] = []
    run = f"g{protocol.score_conditions}_k{protocol.score_samples}"
    for m, n in protocol.settings:
        setting = f"m{m}n{n}"
        experiment = experiments_root / setting
        for model in models:
            variant, training = model_identity(m, n, model)
            base = experiment / "decoder_comparison/linear_cdf" if model == "QWFM" else experiment
            for nfe in (protocol.nfes if model in ("HFM", "QWFM") else (None,)):
                tail = Path(f"nfe_{nfe:04d}") if nfe is not None else Path()
                model_dir = Path(training) / tail
                prefix = {"setting": setting, "model": model, "variant": variant, "nfe": nfe,
                          "decoder": "linear_cdf" if model == "QWFM" else "not_applicable"}
                pr_dir = experiment / training / ("decoder_comparison/linear_cdf" if model == "QWFM" else "evaluation") / tail
                pr_path = pr_dir / "metrics/evaluation_summary.json"
                pr = reports.json(pr_path)
                _require(pr.get("protocol_version"), "saved_samples_reference_oracle_v1", "PR protocol", pr_path)
                _require(pr.get("metric_protocol_version"), "continuous_posterior_metrics_v1", "PR metric protocol", pr_path)
                for key, expected in (("sample_count", protocol.reference_samples), ("reference_repeats", protocol.reference_repeats),
                                      ("nearest_k", protocol.nearest_k), ("w2_num_quantiles", protocol.pr_w2_quantiles)):
                    _require(pr.get("settings", {}).get(key), expected, key, pr_path)
                conditions = pr["per_condition"]
                _require([c["condition_id"] for c in conditions], list(range(protocol.pr_conditions)), "PR condition IDs", pr_path)
                if any(pr.get(key) is None for key in ("primary_reference_run_id", "oracle_reference_run_id")):
                    raise ValueError(f"{pr_path}: missing primary/oracle reference identity.")
                pr_identity = (pr.get("specification_hash"), pr.get("primary_reference_run_id"),
                               pr.get("oracle_reference_run_id"), pr["settings"].get("seed"))
                _consistent(identities, (setting, "pr"), pr_identity, "shared PR conditions/protocol", pr_path)
                specification = _nonempty(pr.get("specification_hash"), "simulator specification", pr_path)
                _consistent(identities, (setting, "specification"), specification, "simulator specification", pr_path)
                sampling_path = pr_dir / "sampling_summary.json"
                sampling = reports.json(sampling_path) if sampling_path.is_file() else {}
                if sampling:
                    _require(sampling.get("checkpoint_iteration"), protocol.checkpoint_iteration, "checkpoint iteration", sampling_path)
                    if "model_subfolder" in sampling:
                        _require(sampling["model_subfolder"], "unet_ema", "EMA model", sampling_path)
                    if "condition_ids" in sampling:
                        _require(sampling["condition_ids"], list(range(protocol.pr_conditions)), "sampled PR condition IDs", sampling_path)
                else:
                    unavailable_metadata.append(f"{setting}/{model}: PR sampling summary does not record checkpoint metadata.")
                if model == "QWFM":
                    _require(sampling.get("decoder_comparison", {}).get("decoder_name"), "linear_cdf", "QWFM PR decoder", sampling_path)
                for distance in ("l2", "w1", "w2"):
                    for metric in ("precision", "recall"):
                        values = [c["aggregate"]["model"]["ground_metrics"][distance][metric]["mean"] for c in conditions]
                        mean, se = _statistics(values, pr["aggregate"]["model"]["ground_metrics"][distance][metric], pr_path)
                        rows.append({**prefix, "metric": metric, "distance": distance, "mean": mean, "standard_error": se,
                                     "condition_count": protocol.pr_conditions, "sample_count": protocol.reference_samples, "report_path": str(pr_path)})
                proper_base = base / "metric_comparisons/proper_scores" / run / "models" / model_dir
                for distance in ("l2", "w2"):
                    for metric in ("energy_score", "kernel_score"):
                        path = proper_base / f"{metric}{'_w2' if distance == 'w2' else ''}.json"
                        report = reports.json(path)
                        draws = 1 if model == "Regression" else protocol.score_samples
                        for key, expected in (("protocol_version", SCORE_PROTOCOLS[metric, distance]), ("available", True),
                                              ("variant", variant), ("nfe", nfe), ("condition_count", protocol.score_conditions),
                                              ("requested_sample_count", protocol.score_samples), ("sample_count", draws)):
                            _require(report.get(key), expected, key, path)
                        _require(report.get("ground_metric", "l2"), distance, "ground distance", path)
                        _require(report.get("specification_hash"), specification, "simulator specification", path)
                        if distance == "w2":
                            _require(report.get("distance_computation"), "exact_piecewise_uniform_inverse_cdf_integration", "W2 computation", path)
                        generation = _generation(report, model, variant, nfe, path, protocol)
                        truth_hash = _nonempty(report.get("source_sha256", {}).get("simulator_truth"), "truth hash", path)
                        sample_hash = _nonempty(report.get("source_sha256", {}).get("samples"), "sample hash", path)
                        _require(generation.get("conditions_sha256"), truth_hash, "generation conditions", path)
                        _consistent(identities, (setting, "truth"), truth_hash, "shared proper-score test set", path)
                        _consistent(identities, (setting, model, nfe, "samples"), sample_hash, "shared saved predictions", path)
                        _consistent(identities, (setting, model, "checkpoint"), generation["checkpoint_identity"]["sha256"], "checkpoint across NFE and metrics", path)
                        conditions = report["per_condition"]
                        _require([c["condition_id"] for c in conditions], list(range(protocol.score_conditions)), "proper-score condition IDs", path)
                        if metric == "kernel_score":
                            calibration = report.get("bandwidth_provenance", {})
                            _require(calibration.get("calibration_count"), protocol.calibration_count, "kernel calibration count", path)
                            _require(calibration.get("calibration_split"), "kernel_score_calibration", "independent kernel calibration", path)
                            _require(calibration.get("evaluation_predictions_used"), False, "kernel calibration independence", path)
                            _require(calibration.get("evaluation_truths_used"), False, "kernel calibration independence", path)
                            bandwidth = report.get("kernel_bandwidth")
                            if not isinstance(bandwidth, (int, float)) or not math.isfinite(bandwidth) or bandwidth <= 0:
                                raise ValueError(f"{path}: invalid kernel bandwidth.")
                            _consistent(identities, (setting, distance, "bandwidth"),
                                        (bandwidth, calibration.get("calibration_histogram_sha256")), "kernel bandwidth/calibration across models", path)
                        mean, se = _statistics([c[metric] for c in conditions], report["aggregate"][metric], path)
                        rows.append({**prefix, "metric": metric, "distance": distance, "mean": mean, "standard_error": se,
                                     "condition_count": protocol.score_conditions, "sample_count": draws, "report_path": str(path)})
                fc_path = base / "metric_comparisons/forward_consistency" / run / "models" / model_dir / "forward_consistency.json"
                fc = reports.json(fc_path)
                _require(fc.get("protocol_version"), "saved_joint_pairs_forward_errors_v1", "forward protocol", fc_path)
                if fc.get("mean_spectrum_protocol_version") != "arithmetic_mean_spectrum_forward_errors_v1":
                    raise ValueError(f"{fc_path}: mean-spectrum errors are absent or outdated. Rerun the offline forward evaluator with --recompute.")
                for key, expected in (("variant", variant), ("nfe", nfe), ("num_measurements", m), ("num_spectrum_bins", n),
                                      ("condition_count", protocol.score_conditions), ("requested_sample_count", protocol.score_samples),
                                      ("sample_count", 1 if model == "Regression" else protocol.score_samples),
                                      ("specification_hash", specification), ("truth_sha256", identities[setting, "truth"]),
                                      ("samples_sha256", identities[setting, model, nfe, "samples"])):
                    _require(fc.get(key), expected, key, fc_path)
                generation = _generation(fc, model, variant, nfe, fc_path, protocol)
                _require(generation["checkpoint_identity"]["sha256"], identities[setting, model, "checkpoint"], "forward checkpoint", fc_path)
                _require(generation.get("conditions_sha256"), fc["truth_sha256"], "forward generation conditions", fc_path)
                csv_path = fc_path.with_name("per_condition.csv")
                conditions = list(csv.DictReader(io.StringIO(reports.read_text(csv_path))))
                _require(reports.hashes[str(csv_path)], fc.get("output_sha256", {}).get("per_condition.csv"), "forward CSV checksum", csv_path)
                _require([int(c["condition_id"]) for c in conditions], list(range(protocol.score_conditions)), "forward condition IDs", csv_path)
                if any("mean_spectrum_clean_normalized_mse" not in c for c in conditions):
                    raise ValueError(f"{csv_path}: missing mean-spectrum errors; rerun the offline forward evaluator with --recompute.")
                if any(float(c["mean_spectrum_clean_normalized_mse"]) > float(c["clean_normalized_mse"]) + 1e-10 for c in conditions):
                    raise ValueError(f"{csv_path}: mean-spectrum squared error exceeds the mean individual error.")
                for field, metric in (("clean_normalized_mse", "forward_individual"),
                                      ("mean_spectrum_clean_normalized_mse", "forward_mean_spectrum")):
                    if field not in fc.get("aggregate", {}) or any(field not in c for c in conditions):
                        raise ValueError(f"{fc_path}: missing {field}; rerun the offline forward evaluator with --recompute.")
                    mean, se = _statistics([float(c[field]) for c in conditions], fc["aggregate"][field], fc_path)
                    rows.append({**prefix, "metric": metric, "distance": "signal_mse_over_sigma_squared", "mean": mean, "standard_error": se,
                                 "condition_count": protocol.score_conditions, "sample_count": fc["sample_count"], "report_path": str(fc_path)})
    return rows, {"source_reports_sha256": reports.hashes, "settings": [list(pair) for pair in protocol.settings],
                  "nfes": list(protocol.nfes), "models": list(models), "qwfm_decoder": "linear_cdf", "quantile_count": "L=N",
                  "network_forward_performed": False, "test_data_generated": False, "forward_errors_recomputed": False,
                  "tensor_files_opened": False,
                  "unavailable_metadata": unavailable_metadata,
                  "validation_notes": "Report hashes and cross-report identities checked; source tensors are not opened. Checkpoint iteration is checked in PR sampling summaries and wherever recorded; EMA identity is checked in proper-score and forward reports."}


def distance_table(rows: list[dict]) -> list[dict]:
    """Table 1: F1 of unrounded aggregate P/R; no W1 proper scores."""
    selected = [r for r in rows if r["setting"] == "m64n128" and r["nfe"] in (None, 256)
                and r["metric"] in ("precision", "recall", "energy_score", "kernel_score")]
    index = {(r["model"], r["metric"], r["distance"]): r for r in selected}
    for model in MODELS:
        for distance in ("l2", "w1", "w2"):
            if (model, "precision", distance) not in index:
                continue
            p, r = index[model, "precision", distance], index[model, "recall", distance]
            selected.append({**p, "metric": "f1", "mean": 2*p["mean"]*r["mean"]/(p["mean"]+r["mean"]) if p["mean"]+r["mean"] else 0.0,
                             "standard_error": None})
    order = {name: i for i, name in enumerate(("precision", "recall", "f1", "energy_score", "kernel_score"))}
    return sorted(selected, key=lambda r: (order[r["metric"]], ("l2", "w1", "w2").index(r["distance"]), MODELS.index(r["model"])))


def forward_table(rows: list[dict]) -> list[dict]:
    index = {(r["setting"], r["model"], r["nfe"]): r for r in rows if r["metric"] == "forward_mean_spectrum"}
    return [{"setting": r["setting"], "model": r["model"], "nfe": r["nfe"], "individual_fc": r["mean"],
             "individual_standard_error": r["standard_error"], "mean_spectrum_fc": index[r["setting"], r["model"], r["nfe"]]["mean"],
             "mean_spectrum_standard_error": index[r["setting"], r["model"], r["nfe"]]["standard_error"],
             "condition_count": r["condition_count"], "sample_count": r["sample_count"]}
            for r in rows if r["metric"] == "forward_individual"]


def plot_figure2(rows: list[dict], settings: tuple[tuple[int, int], ...], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("precision", "recall", "energy_score", "kernel_score")
    labels = ("Precision", "Recall", "Energy score", "Kernel score")
    colors = {"HFM": "#0072B2", "QWFM": "#D55E00", "Regression": "#009E73", "CVAE": "#CC79A7"}
    fig, axes = plt.subplots(4, len(settings), squeeze=False, sharey="row", figsize=(4*len(settings), 9), constrained_layout=True)
    for column, (m, n) in enumerate(settings):
        for row, metric in enumerate(metrics):
            ax = axes[row, column]
            for model in MODELS:
                values = sorted((r for r in rows if r["setting"] == f"m{m}n{n}" and r["model"] == model
                                 and r["metric"] == metric and r["distance"] == "w2"), key=lambda r: r["nfe"] or 0)
                if model in ("HFM", "QWFM"):
                    ax.errorbar([r["nfe"] for r in values], [r["mean"] for r in values],
                                yerr=[r["standard_error"] for r in values], color=colors[model], marker="o", markersize=4, capsize=2, label=model)
                else:
                    value = values[0]
                    ax.axhline(value["mean"], color=colors[model], linestyle="--" if model == "Regression" else "-.", label=model)
                    ax.axhspan(value["mean"]-value["standard_error"], value["mean"]+value["standard_error"], color=colors[model], alpha=.12)
            ax.set_xscale("log", base=4)
            ax.set_xticks((1, 4, 16, 64, 256), labels=("1", "4", "16", "64", "256"))
            ax.grid(alpha=.2)
            if column == 0:
                ax.set_ylabel(labels[row])
            if row == 0:
                ax.set_title(f"M={m}, N={n}, L={n}")
            if row == 3:
                ax.set_xlabel("NFE")
    axes[0, 0].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=240, facecolor="white")
    plt.close(fig)


def export_results(experiments_root: Path, output_dir: Path, protocol: PaperProtocol = PaperProtocol(), *, plots: bool = True) -> dict:
    rows, manifest = collect_reports(experiments_root, protocol)
    # Validate all inputs before creating any output files.
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"all_metrics.csv": rows, "table1_distance_sensitivity.csv": distance_table(rows),
              "table2_forward_consistency.csv": forward_table(rows)}
    for name, records in tables.items():
        if not records:
            continue
        with (output_dir/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if plots:
        plot_figure2(rows, protocol.settings, output_dir/"figure2_metrics_nfe")
    outputs = [output_dir/name for name, records in tables.items() if records]
    if plots:
        outputs.extend(output_dir/f"figure2_metrics_nfe.{extension}" for extension in ("png", "pdf"))
    manifest.update(protocol_version="paper_results_report_export_v1", metric_row_count=len(rows),
                    outputs_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
                    figure2_error_bars="one standard error across independent test conditions",
                    table1_f1="harmonic mean of unrounded aggregate precision and recall",
                    table1_included=bool(tables["table1_distance_sensitivity.csv"]))
    (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiments_root", type=Path, default=Path("outputs"), help="Directory containing m32n128, m64n128 and m128n256.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--settings", nargs="+", choices=[f"m{m}n{n}" for m, n in SETTINGS],
                        help="Optional subset of the paper settings; Table 1 requires m64n128.")
    parser.add_argument("--skip_plots", action="store_true", help="Export CSV tables and provenance without rendering Figure 2.")
    args = parser.parse_args(argv)
    settings = tuple(pair for pair in SETTINGS if args.settings is None or f"m{pair[0]}n{pair[1]}" in args.settings)
    try:
        manifest = export_results(args.experiments_root.expanduser().resolve(), args.output_dir.expanduser().resolve(),
                                  PaperProtocol(settings=settings), plots=not args.skip_plots)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Evaluation report schema is incomplete or incompatible: {error}") from error
    print(f"Exported {manifest['metric_row_count']} metric rows to {args.output_dir} (reports only).")
    return manifest


if __name__ == "__main__":
    main()
