"""Visualize selected models from one setting's saved reconstruction samples.

Works with a single completed model or a comparison within one setting. No
checkpoint, neural inference, SMC bank, or samples from other models are needed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import torch

from codes.evaluate.visualize_paper_reconstructions import (  # noqa: E402
    SELECTION, _load_verified_input, collect_display_arrays,
    experiment_dimensions, select_cases, validate_truth,
)
from codes.utils.io_utils import atomic_write_json  # noqa: E402
from codes.utils.predictive_generation_utils import file_sha256  # noqa: E402
from codes.utils.proper_score_locks import file_lock  # noqa: E402

MODEL_NAMES = ("HFM", "QWFM", "Regression", "CVAE")
COLORS = {"HFM": "#0072B2", "QWFM": "#D55E00", "Regression": "#009E73", "CVAE": "#CC79A7"}


def render_case(arrays: dict, case: dict, index: int, output_dir: Path,
                dimensions: tuple[int, int], models: tuple[str, ...], nfe: int,
                num_quantiles: int) -> dict:
    """One decay/spectrum pair per model; every saved draw stays visible."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(len(models), 2, squeeze=False, sharex="col", sharey="col",
                                figsize=(11, 3.15 * len(models) + .55), constrained_layout=True)
    times, edges = arrays["echo_times_ms"], arrays["rate_edges"]
    signal_values = [arrays["observation"][index],
                     *(arrays[f"{label}_shared_noise_signal"][index] for label in models)]
    signal_min = min(0., *(float(value.min()) for value in signal_values))
    signal_max = max(1., *(float(value.max()) for value in signal_values))
    signal_margin = .025 * max(signal_max - signal_min, 1e-8)
    spectrum_max = 1.08 * max(float(arrays["truth_histogram"][index].max()),
                              *(float(arrays[f"{label}_histogram"][index].max()) for label in models))
    counts = {}
    for row, model in enumerate(models):
        decay, spectrum = axes[row]
        signals = arrays[f"{model}_shared_noise_signal"][index]
        histograms = arrays[f"{model}_histogram"][index]
        count = len(histograms)
        counts[model] = count
        color = COLORS[model]
        label = "Prediction" if count == 1 else f"{count} saved draws"
        # Plot in stored order without smoothing, sorting, or selecting draws.
        for draw, (signal, histogram) in enumerate(zip(signals, histograms)):
            kwargs = {"color": color, "alpha": 1. if count == 1 else .45,
                      "linewidth": 1.4 if count == 1 else .9,
                      "label": label if draw == 0 else None}
            decay.plot(times, signal, **kwargs)
            spectrum.stairs(histogram, edges, baseline=None, **kwargs)
        if count > 1:
            decay.plot(times, signals.mean(axis=0), color=color, linewidth=1.7,
                       linestyle="--", label="Sample mean")
            spectrum.stairs(histograms.mean(axis=0), edges, baseline=None, color=color,
                            linewidth=1.7, linestyle="--", label="Sample mean")
        decay.plot(times, arrays["observation"][index], color="#222222", linestyle=":",
                   linewidth=1.6, label="Observed decay")
        spectrum.stairs(arrays["truth_histogram"][index], edges, baseline=None,
                        color="#222222", linestyle=":", linewidth=1.6, label="Ground truth")
        suffix = f", NFE={nfe}" if model in ("HFM", "QWFM") else ", NFE-independent"
        if model == "QWFM":
            suffix += f", L={num_quantiles}"
        decay.set_title(f"{model}{suffix}: decay", loc="left", fontsize=10)
        spectrum.set_title(f"{model}: spectrum", loc="left", fontsize=10)
        decay.set_ylabel("Normalized signal")
        spectrum.set_ylabel("Bin mass")
        decay.set_ylim(signal_min - signal_margin, signal_max + signal_margin)
        spectrum.set_ylim(0, spectrum_max)
        for ax, values in ((decay, times), (spectrum, edges)):
            ax.set_xscale("log")
            ax.set_xlim(values[0], values[-1])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(alpha=.18, which="major")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8, frameon=False, loc="best")
    axes[-1, 0].set_xlabel("Echo time (ms)")
    axes[-1, 1].set_xlabel(r"Relaxation rate ($s^{-1}$)")
    figure.suptitle(f"{case['morphology']} · condition {case['condition_id']} · "
                   f"M={dimensions[0]}, N={dimensions[1]}", fontsize=12)
    name = f"m{dimensions[0]}n{dimensions[1]}_{case['slug']}"
    paths = {extension: output_dir / f"{name}.{extension}" for extension in ("png", "pdf")}
    for extension, path in paths.items():
        figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    return {"condition_id": case["condition_id"], "models": list(models),
            "saved_draw_counts": counts, "files": {key: str(path) for key, path in paths.items()},
            "sha256": {key: file_sha256(path) for key, path in paths.items()},
            "spectral_ylim": [0, spectrum_max],
            "signal_ylim": [signal_min - signal_margin, signal_max + signal_margin]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--models", choices=MODEL_NAMES, nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--nfe", type=int, default=256)
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--num_quantiles", type=int, choices=(128, 256, 512))
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.nfe, args.num_conditions, args.sample_count, args.num_cpu_threads) < 1:
        parser.error("NFE, condition count, sample count, and CPU thread count must be positive.")
    if len(set(args.models)) != len(args.models):
        parser.error("Select each model at most once.")
    args.models = tuple(args.models)
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dimensions = experiment_dimensions(args.experiment_dir)
    args.num_quantiles = args.num_quantiles or args.dimensions[1]
    if args.num_quantiles not in (128, 256, 512):
        parser.error("Specify a supported --num_quantiles value (128, 256, or 512).")
    run_name = f"g{args.num_conditions}_k{args.sample_count}"
    args.run_dir = args.experiment_dir / "proper_scores" / run_name
    linear_run = args.experiment_dir / "decoder_comparison/linear_cdf/proper_scores" / run_name
    for source in (args.run_dir.resolve(), linear_run.resolve()):
        if args.output_dir == source or source in args.output_dir.parents:
            parser.error("Write figures outside the saved-input directories.")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(args.num_cpu_threads)
    truth_path = args.run_dir / "test_conditions.pt"
    truth, truth_hash = _load_verified_input(truth_path)
    spec, checks = validate_truth(truth, args.num_conditions)
    if (spec.num_measurements, spec.num_spectrum_bins) != args.dimensions:
        raise ValueError("Experiment dimensions differ from the saved truths.")
    # Model subsets use exactly the same truth-only selection as comparisons.
    selected = select_cases(truth)
    arrays, sources = collect_display_arrays(
        truth, truth_hash, selected, args.experiment_dir, args.nfe, args.sample_count,
        args.num_quantiles, models=args.models)
    # Validate every requested source before publishing any figures.
    with file_lock(args.output_dir / ".visualization.lock", blocking=False):
        array_path = args.output_dir / "selected_display_arrays.npz"
        np.savez_compressed(array_path, **arrays)
        figures = [render_case(arrays, case, index, args.output_dir, args.dimensions,
                               args.models, args.nfe, args.num_quantiles)
                   for index, case in enumerate(selected)]
        case_path = args.output_dir / "selected_cases.csv"
        with case_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        record = {
            "protocol": "saved_reconstruction_model_comparison_v1",
            "models": list(args.models), "dimensions": {"M": args.dimensions[0], "N": args.dimensions[1]},
            "flow_nfe": args.nfe if any(model in ("HFM", "QWFM") for model in args.models) else None,
            "num_quantiles": args.num_quantiles if "QWFM" in args.models else None,
            "sample_count": args.sample_count, "selection": SELECTION,
            "selected_cases": selected, "figures": figures,
            "truth_path": str(truth_path), "truth_sha256": truth_hash,
            "truth_consistency_checks": checks, "sources": sources,
            "display_arrays": str(array_path), "display_arrays_sha256": file_sha256(array_path),
            "selected_cases_csv": str(case_path), "selected_cases_csv_sha256": file_sha256(case_path),
            "signal_formulas": {"reference": "y = A h + epsilon", "prediction": "A h_hat + epsilon",
                                "original_noise": "epsilon = y - A h (recovered in float64)"},
            "same_original_noise_reused": True,
            "max_abs_recovered_minus_saved_original_noise": float(np.abs(
                arrays["original_noise"] - arrays["saved_original_noise"]).max()),
            "display": "One decay/spectrum row per model; all draws in saved order, with the sample mean "
                       "and ground-truth reference overlaid. Common axes; no smoothing or draw selection.",
            "no_model_inference": True, "no_test_simulation": True, "no_smc": True,
            "script_sha256": file_sha256(Path(__file__)),
        }
        atomic_write_json(args.output_dir / "qualitative_provenance.json", record)
    print(f"Saved {len(figures)} reconstruction plots for {', '.join(args.models)} to {args.output_dir}", flush=True)
    return record


if __name__ == "__main__":
    main()
