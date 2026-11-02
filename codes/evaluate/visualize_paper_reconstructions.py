"""Render five morphology-selected reconstruction pages from saved predictions.

QWFM reads linear-CDF samples; HFM, CVAE, and regression read their ordinary
proper-score samples. Every stochastic draw is retained in saved order. This
offline reader requires no manuscript files, model weights, or comparison review.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.signal import find_peaks
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from codes.evaluate.evaluate_forward_consistency import (  # noqa: E402
    _load_verified_input, validate_samples, validate_truth,
)
from codes.metrics.forward_consistency import per_sample_forward_errors  # noqa: E402
from codes.utils.io_utils import atomic_write_json  # noqa: E402
from codes.utils.model_variant_utils import FLOW_METHODS, VARIANTS, experiment_dimensions  # noqa: E402
from codes.utils.predictive_generation_utils import file_sha256  # noqa: E402
from codes.utils.proper_score_locks import file_lock  # noqa: E402
from codes.utils.nmr_utils import decode_quantiles, make_normalized_log_rate_edges  # noqa: E402
from codes.utils.quantile_decoder_artifacts import validate_saved_quantiles  # noqa: E402

MODELS = (
    ("histogram_linear", "HFM", "#0072B2"),
    ("quantile_wasserstein_l128", "QWFM", "#D55E00"),
    ("conditional_vae_z16", "CVAE", "#7B61A8"),
    ("deterministic_regression_mse_w1", "Regression", "#009E73"),
)
CATEGORIES = (
    ("broad_single", "Broad single peak", 1, .90),
    ("two_peaks", "Two separated peaks", 2, .50),
    ("three_peaks", "Three separated peaks", 3, .50),
    ("four_peaks", "Four separated peaks", 4, .50),
    ("five_peaks", "Five separated peaks", 5, .50),
)
SELECTION = (
    "Selection uses saved ground-truth histograms and mixture parameters only, before "
    "opening model predictions. Require matching component and resolved-peak counts; "
    "peak height and prominence are at least 5% of the largest bin mass. For multiple "
    "components require every mixture weight >=0.10 and adjacent peak separation "
    ">=0.40 in natural-log rate. Select the 90th percentile of log-rate standard "
    "deviation for a broad single peak and the median for each multiple-peak group. Sort ties "
    "by condition ID and use round(q*(candidate_count-1))."
)


def models_for_quantiles(num_quantiles: int) -> tuple:
    """Select the saved QWFM variant without changing the comparison models."""
    variant = f"quantile_wasserstein_l{num_quantiles}"
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported QWFM quantile count: {num_quantiles}.")
    return (MODELS[0], (variant, *MODELS[1][1:]), *MODELS[2:])


def select_cases(truth: dict) -> list[dict]:
    """Fix one case per peak count using only truths; never inspect model errors."""
    histogram = truth["histogram"].double().numpy()
    probability = histogram / histogram.sum(axis=1, keepdims=True)
    edges = truth["grid"]["log_rate_edges"].double().numpy()
    centers, widths = .5 * (edges[:-1] + edges[1:]), np.diff(edges)
    mean = probability @ centers
    spread = np.sqrt(np.maximum(probability @ (centers**2 + widths**2 / 12) - mean**2, 0))
    counts = truth["component_count"].numpy()
    active = truth["active_mask"].numpy()
    weights = truth["mixture_weights"].numpy()
    features = []
    for index, values in enumerate(histogram):
        peaks, _ = find_peaks(values, height=.05 * values.max(), prominence=.05 * values.max())
        features.append({
            "index": index, "condition_id": int(truth["sample_id"][index]),
            "component_count": int(counts[index]), "peak_count": len(peaks),
            "log_rate_sd": float(spread[index]),
            "minimum_peak_separation": float(np.diff(centers[peaks]).min()) if len(peaks) > 1 else 0.,
            "minimum_weight": float(weights[index, active[index]].min()),
        })
    selected = []
    for slug, title, count, quantile in CATEGORIES:
        candidates = [row for row in features if row["component_count"] == count and row["peak_count"] == count]
        if count > 1:
            candidates = [row for row in candidates if row["minimum_weight"] >= .10
                          and row["minimum_peak_separation"] >= .40]
        candidates.sort(key=lambda row: (row["log_rate_sd"], row["condition_id"]))
        if not candidates:
            raise ValueError(f"No ground-truth candidate satisfies morphology: {title}.")
        rank = round(quantile * (len(candidates) - 1))
        selected.append({**candidates[rank], "slug": slug, "morphology": title,
                         "candidate_count": len(candidates), "selected_quantile": quantile, "rank": rank})
    return selected


def prediction_path(experiment_dir: Path, run_name: str, variant: str,
                    dimensions: tuple[int, int], nfe: int) -> Path:
    """Route QWFM to its linear-CDF cache without reading legacy predictions."""
    method, pattern, _, _ = VARIANTS[variant]
    base = experiment_dir
    if method == "quantile_wasserstein":
        base = base / "decoder_comparison" / "linear_cdf"
    path = base / "proper_scores" / run_name / pattern.format(
        shape=f"m{dimensions[0]}_n{dimensions[1]}")
    if method in FLOW_METHODS:
        path /= f"nfe_{nfe:04d}"
    return path / "samples.pt"


def collect_display_arrays(truth: dict, truth_hash: str, selected: list[dict],
                           experiment_dir: Path, nfe: int, sample_count: int,
                           num_quantiles: int = 128, *,
                           models: tuple[str, ...] | list[str] | None = None) -> tuple[dict, dict]:
    """Read validated saved draws and reuse the observation's original noise."""
    available_models = {record[1]: record for record in models_for_quantiles(num_quantiles)}
    chosen_models = tuple(available_models) if models is None else tuple(models)
    if not chosen_models or len(set(chosen_models)) != len(chosen_models):
        raise ValueError("Select at least one model without duplicates.")
    if any(label not in available_models for label in chosen_models):
        raise ValueError(f"Unknown reconstruction model; choose from {tuple(available_models)}.")
    indices = [row["index"] for row in selected]
    arrays = {
        "condition_ids": truth["sample_id"][indices].numpy().copy(),
        "truth_histogram": truth["histogram"][indices].double().numpy().copy(),
        "observation": truth["measurement"][indices].double().numpy().copy(),
        "rate_edges": truth["grid"]["rate_edges"].double().numpy().copy(),
        "echo_times_ms": 1000 * truth["grid"]["echo_times"].double().numpy().copy(),
    }
    kernel = truth["grid"]["kernel"].double()
    histograms = truth["histogram"][indices]
    observations = truth["measurement"][indices]
    sigma = truth["specification"]["noise_standard_deviation"]
    dimensions = (kernel.shape[0], kernel.shape[1])
    run_name = f"g{len(truth['sample_id'])}_k{sample_count}"
    arrays["clean_signal"] = (histograms.double() @ kernel.T).numpy()
    # Recover epsilon in float64 to retain y=A h+epsilon despite storage rounding.
    arrays["original_noise"] = arrays["observation"] - arrays["clean_signal"]
    arrays["saved_original_noise"] = truth["noise"][indices].double().numpy().copy()
    sources = {}
    for variant, label, _ in (available_models[label] for label in chosen_models):
        method = VARIANTS[variant][0]
        steps = nfe if method in FLOW_METHODS else None
        path = prediction_path(experiment_dir, run_name, variant, dimensions, nfe)
        artifact, digest = _load_verified_input(path)
        provenance = validate_samples(artifact, truth, truth_hash, variant, steps, sample_count)
        samples = artifact["samples"][indices].clone()
        decoder = None
        if method == "quantile_wasserstein":
            quantiles = validate_saved_quantiles(artifact)
            decoder = artifact["decoder_comparison"]
            if decoder["decoder_name"] != "linear_cdf":
                raise ValueError("Paper reconstruction figures require linear_cdf QWFM samples.")
            if quantiles.shape[-1] != num_quantiles:
                raise ValueError("Saved QWFM quantile count differs from the requested variant.")
            edges = make_normalized_log_rate_edges(artifact["grid"]["log_rate_edges"])
            decoded = decode_quantiles(quantiles[indices].reshape(-1, num_quantiles),
                                       edges, decoder="linear_cdf").reshape_as(samples)
            if not torch.equal(decoded.to(samples.dtype), samples):
                raise ValueError("Displayed QWFM histograms do not match their saved linear-CDF quantiles.")
        errors = per_sample_forward_errors(samples, histograms, observations, kernel,
                                           noise_standard_deviation=sigma)
        prediction = samples.double() @ kernel.T
        arrays[f"{label}_histogram"] = samples.double().numpy()
        arrays[f"{label}_forward_signal"] = prediction.numpy()
        arrays[f"{label}_shared_noise_signal"] = prediction.numpy() + arrays["original_noise"][:, None, :]
        for name, values in errors.items():
            arrays[f"{label}_{name}"] = values.numpy()
        sources[label] = {
            "variant": variant, "samples_path": str(path), "samples_sha256": digest,
            "saved_draw_indices": list(range(samples.shape[1])), "nfe": steps,
            "generation_provenance": provenance,
        }
        if decoder is not None:
            sources[label]["decoder"] = decoder
            sources[label]["displayed_histograms_match_saved_quantiles"] = True
    return arrays, sources


def render_case(arrays: dict, case: dict, row: int, output: Path, dimensions: tuple[int, int],
                sigma: float, nfe: int, num_quantiles: int = 128) -> dict:
    """Show horizontal decay/spectrum pairs with each reference displayed once."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .45,
        "savefig.facecolor": "white", "xtick.labelsize": 5.2, "ytick.labelsize": 5.2,
    })
    figure = plt.figure(figsize=(7.2, 7.6))
    edges, times = arrays["rate_edges"], arrays["echo_times_ms"]
    spectrum_max = 1.07 * max(arrays["truth_histogram"][row].max(),
                              *(arrays[f"{label}_histogram"][row].max() for _, label, _ in MODELS))
    displayed_signals = [arrays["observation"][row],
                         *(arrays[f"{label}_shared_noise_signal"][row] for _, label, _ in MODELS)]
    signal_min = min(0., *(float(signal.min()) for signal in displayed_signals))
    signal_max = max(1., *(float(signal.max()) for signal in displayed_signals))
    signal_limits = (signal_min - .015, signal_max + .015)
    left, right = .047, .991
    shot_gap, pair_gap = .026, .013
    shot_width = (right - left - 3 * shot_gap) / 4

    def axis(rect, *, signal=False, left_labels=False, bottom_labels=False):
        ax = figure.add_axes(rect)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(length=1.8, pad=1.2, width=.4)
        if signal:
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(*signal_limits)
            ax.set_xticks([times[0], times[-1]],
                         [f"{times[0]:g}", f"{times[-1]:g}"] if bottom_labels else [])
            ax.set_yticks([0, 1], ["0", "1"])
        else:
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, spectrum_max)
            ax.set_xticks([edges[0], edges[-1]],
                         [f"{edges[0]:g}", f"{edges[-1]:g}"] if bottom_labels else [])
            ax.set_yticks([0, spectrum_max / 2], ["0", f"{spectrum_max / 2:.2g}"])
        if not left_labels:
            ax.tick_params(labelleft=False)
        if bottom_labels:
            ax.get_xticklabels()[0].set_ha("left")
            ax.get_xticklabels()[-1].set_ha("right")
        return ax

    def pair(x, top, width, signal, histogram, color, *, observed=False,
             left_labels=False, bottom_labels=False, height=.068, gap=pair_gap):
        plot_width = (width - gap) / 2
        decay = axis([x, top-height, plot_width, height], signal=True,
                     left_labels=left_labels, bottom_labels=bottom_labels)
        if observed:
            decay.plot(times, signal, ".", color=color, ms=2.0)
        else:
            decay.plot(times, signal, color=color, lw=.75)
        spectrum = axis([x+plot_width+gap, top-height, plot_width, height],
                        left_labels=left_labels, bottom_labels=bottom_labels)
        spectrum.stairs(histogram, edges, baseline=None, color=color, lw=.7)
        for ax, title in [(decay, "Decay"), (spectrum, "Spectrum")]:
            ax.set_title(title, fontsize=5.5, pad=2, color="#555555")
        return decay, spectrum

    figure.text(left, .989, case["morphology"], va="top", fontsize=10, weight="bold")
    figure.text(right, .989, f"Condition {case['condition_id']}  |  M={dimensions[0]}, N={dimensions[1]}",
                va="top", ha="right", fontsize=7, color="#555555")
    # Center the reference and regression pairs in the middle two shot slots.
    # Every reference and generated panel has exactly the same width, height,
    # and corresponding axis limits for direct shape comparisons.
    references = [
        ("Reference", arrays["observation"][row], arrays["truth_histogram"][row], "#232323", True),
        ("Regression", arrays["Regression_shared_noise_signal"][row, 0],
         arrays["Regression_histogram"][row, 0], "#009E73", False),
    ]
    for index, (title, signal, histogram, color, observed) in enumerate(references):
        x = left + (index + 1) * (shot_width + shot_gap)
        figure.text(x + shot_width / 2, .953, title, va="bottom", ha="center",
                    fontsize=8.4, weight="bold", color=color)
        pair(x, .918, shot_width, signal, histogram, color, observed=observed,
             bottom_labels=True)

    # Stack HFM, QWFM, and CVAE vertically. Each model occupies two rows of
    # four horizontal decay/spectrum pairs, giving eight plot columns.
    for model_index, (_, label, color) in enumerate(MODELS[:3]):
        model_top = .748 - .264 * model_index
        method_title = label + (rf" ($L={num_quantiles}$)" if label == "QWFM" else r" ($z=16$)" if label == "CVAE" else "")
        figure.text(left, model_top+.053, method_title, va="bottom", ha="left",
                    fontsize=8.4, weight="bold", color=color)
        divider = model_top + .073
        figure.add_artist(plt.Line2D([left, right], [divider, divider],
                                    transform=figure.transFigure, color="#d6d6d6", lw=.5))
        for draw in range(8):
            x = left + (draw % 4) * (shot_width + shot_gap)
            top = model_top - .110 * (draw // 4)
            figure.text(x + shot_width / 2, top+.024, f"Draw {draw+1}", fontsize=5.8,
                        ha="center", va="bottom", color=color)
            pair(x, top, shot_width, arrays[f"{label}_shared_noise_signal"][row, draw],
                 arrays[f"{label}_histogram"][row, draw], color)

    # Save both formats directly; no external PDF converter is needed.
    slug = f"m{dimensions[0]}n{dimensions[1]}_{case['slug']}_eight_draws"
    pdf, png = output / f"{slug}.pdf", output / f"{slug}.png"
    figure.savefig(pdf, bbox_inches="tight", pad_inches=.04,
                   metadata={"Title": f"{case['morphology']}: observed and reconstructed signal/spectrum pairs"})
    figure.savefig(png, dpi=220, bbox_inches="tight", pad_inches=.04)
    plt.close(figure)
    return {"pdf": str(pdf), "png": str(png), "pdf_sha256": file_sha256(pdf), "png_sha256": file_sha256(png),
            "condition_id": case["condition_id"], "size_inches": [7.2, 7.6],
            "num_quantiles": num_quantiles,
            "stochastic_samples": 24, "regression_predictions": 1, "ground_truth_spectra": 1,
            "spectral_ylim": [0, float(spectrum_max)], "signal_ylim": list(signal_limits),
            "pair_order": ["signal", "spectrum"],
            "pair_orientation": "horizontal",
            "reference_display_repetitions": 1, "regression_display_repetitions": 1,
            "reference_layout": "centered row of four panels, identical size to generated panels: reference decay/spectrum, regression decay/spectrum",
            "stochastic_layout": "HFM, QWFM, CVAE stacked vertically, each with 2 rows of 4 draws (8 plot columns)"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment_dir", type=Path, default=Path("outputs/m64n128"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper/figures/reconstructions"))
    parser.add_argument("--nfe", type=int, default=256)
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--num_quantiles", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.sample_count != 8:
        parser.error("These pages display exactly eight saved stochastic draws.")
    if min(args.nfe, args.num_conditions, args.num_cpu_threads) < 1:
        parser.error("NFE, condition count, and CPU thread count must be positive.")
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dimensions = experiment_dimensions(args.experiment_dir)
    args.run_dir = args.experiment_dir / "proper_scores" / f"g{args.num_conditions}_k8"
    linear_run = (args.experiment_dir / "decoder_comparison" / "linear_cdf" /
                  "proper_scores" / args.run_dir.name)
    for input_dir in (args.run_dir.resolve(), linear_run.resolve()):
        if args.output_dir == input_dir or input_dir in args.output_dir.parents:
            parser.error("Write figures outside the saved-input directories.")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(args.num_cpu_threads)
    truth_path = args.run_dir / "test_conditions.pt"
    truth, truth_hash = _load_verified_input(truth_path)
    spec, consistency = validate_truth(truth, args.num_conditions)
    if (spec.num_measurements, spec.num_spectrum_bins) != args.dimensions:
        raise ValueError("Experiment dimensions differ from the saved truths.")
    # Complete truth-only selection before opening any model prediction.
    selected = select_cases(truth)
    arrays, sources = collect_display_arrays(
        truth, truth_hash, selected, args.experiment_dir, args.nfe,
        args.sample_count, args.num_quantiles)
    with file_lock(args.output_dir / ".visualization.lock", blocking=False):
        array_path = args.output_dir / "selected_display_arrays.npz"
        np.savez_compressed(array_path, **arrays)
        figures = [render_case(arrays, case, row, args.output_dir, args.dimensions,
                               spec.noise_standard_deviation, args.nfe, args.num_quantiles)
                   for row, case in enumerate(selected)]
        csv_path = args.output_dir / "selected_cases.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        record = {
            "protocol": "saved_linear_cdf_reconstruction_pages_v1",
            "dimensions": {"M": args.dimensions[0], "N": args.dimensions[1]},
            "num_quantiles": args.num_quantiles, "nfe": args.nfe,
            "sample_count": args.sample_count, "selection": SELECTION,
            "selected_cases": selected, "figures": figures,
            "truth_path": str(truth_path), "truth_sha256": truth_hash,
            "truth_consistency_checks": consistency, "sources": sources,
            "display_arrays": str(array_path), "display_arrays_sha256": file_sha256(array_path),
            "selected_cases_csv": str(csv_path), "selected_cases_csv_sha256": file_sha256(csv_path),
            "signal_formulas": {"reference": "y = A h + epsilon",
                                "prediction": "A h_hat + epsilon",
                                "original_noise": "epsilon = y - A h (recovered in float64)"},
            "same_original_noise_reused": True,
            "max_abs_recovered_minus_saved_original_noise": float(np.abs(
                arrays["original_noise"] - arrays["saved_original_noise"]).max()),
            "display": "One reference and one regression pair; eight original-order draws each "
                       "for HFM, linear-CDF QWFM, and CVAE. Common axes; no smoothing or draw selection.",
            "no_model_inference": True, "no_test_simulation": True, "no_smc": True,
            "script_sha256": file_sha256(Path(__file__)),
        }
        atomic_write_json(args.output_dir / "qualitative_provenance.json", record)
    print(f"Saved {len(figures)} reconstruction pages to {args.output_dir}", flush=True)
    return record


if __name__ == "__main__":
    main()
