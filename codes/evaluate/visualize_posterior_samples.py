# ruff: noqa: E402
"""Visualize cached reference and model posterior samples without inference."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.evaluate.artifact_contract import (
    NORMALIZED_HISTOGRAM_REPRESENTATION,
    RAW_FORWARD_SPECTRUM_REPRESENTATION,
    validate_optional_forward_spectra,
)
from codes.metrics import derived_quantities
from codes.utils.io_utils import atomic_write_json, load_torch_artifact


VISUALIZATION_PROTOCOL_VERSION = "continuous_posterior_kshot_gallery_v2"
_GRID_FIELDS = (
    "log_rate_edges",
    "rate_edges",
    "rate_centers",
    "echo_times_64",
    "echo_subset_indices",
    "echo_times",
    "kernel",
)


def _same_tensor(first: torch.Tensor, second: torch.Tensor, *, atol: float) -> bool:
    if tuple(first.shape) != tuple(second.shape):
        return False
    if first.is_floating_point() or second.is_floating_point():
        return bool(
            torch.allclose(
                first.detach().to(device="cpu", dtype=torch.float64),
                second.detach().to(device="cpu", dtype=torch.float64),
                rtol=0.0,
                atol=atol,
            )
        )
    return bool(torch.equal(first.detach().cpu(), second.detach().cpu()))


def _validate_simplex(samples: torch.Tensor, *, name: str) -> None:
    if not isinstance(samples, torch.Tensor) or samples.ndim != 3:
        raise ValueError(f"{name}.samples must have shape [G,K,N].")
    if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
        raise ValueError(f"{name}.samples must be finite floating point.")
    if bool((samples < 0).any()):
        raise ValueError(f"{name}.samples contains negative mass.")
    mass = samples.sum(dim=-1)
    if not bool(torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5)):
        raise ValueError(f"{name}.samples contains non-unit histogram mass.")


def validate_visualization_artifacts(
    reference: dict[str, Any],
    model_artifacts: dict[str, dict[str, Any]],
    *,
    qualitative_sample_count: int = 16,
) -> ContinuousNMRSpec:
    """Validate model/reference identities and grids before plotting."""

    if not isinstance(reference, dict):
        raise ValueError("reference artifact must be a dictionary.")
    if not model_artifacts:
        raise ValueError("at least one labelled model artifact is required.")
    if qualitative_sample_count != 16:
        raise ValueError(
            "this protocol fixes qualitative_sample_count=16 (a 4x4 sheet)."
        )
    if "specification" not in reference:
        raise ValueError("reference artifact is missing specification.")
    spec = ContinuousNMRSpec.from_dict(reference["specification"])
    if reference.get("protocol_id") != spec.protocol_id:
        raise ValueError("reference protocol_id differs from its specification.")
    if reference.get("specification_hash") != spec.specification_hash:
        raise ValueError("reference specification hash is inconsistent.")
    expected_grid = build_nmr_grid(spec).as_artifact()
    reference_grid = reference.get("grid")
    if not isinstance(reference_grid, dict):
        raise ValueError("reference artifact is missing its grid.")
    for field in _GRID_FIELDS:
        if field not in reference_grid or not isinstance(
            reference_grid[field], torch.Tensor
        ):
            raise ValueError(f"reference grid is missing tensor {field!r}.")
        expected = expected_grid[field]
        tolerance = 0.0 if not expected.is_floating_point() else 1e-12
        if not _same_tensor(reference_grid[field], expected, atol=tolerance):
            raise ValueError(f"reference grid field {field!r} differs from the spec.")
    if reference_grid.get("specification_hash") != spec.specification_hash:
        raise ValueError("reference grid specification hash is inconsistent.")

    condition_ids = reference.get("condition_ids")
    measurements = reference.get("measurements")
    reference_histograms = reference.get("reference_histograms")
    truth = reference.get("truth_histograms")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (condition_ids, measurements, reference_histograms)
    ):
        raise ValueError(
            "reference identity, measurement, and histogram fields are required."
        )
    groups = int(condition_ids.shape[0])
    if condition_ids.ndim != 1 or int(torch.unique(condition_ids).numel()) != groups:
        raise ValueError("reference condition_ids must be a unique vector.")
    if tuple(measurements.shape) != (groups, spec.num_measurements):
        raise ValueError("reference measurements have the wrong shape.")
    if (
        reference_histograms.ndim != 3
        or reference_histograms.shape[0] != groups
        or reference_histograms.shape[1] < qualitative_sample_count
        or reference_histograms.shape[2] != spec.num_spectrum_bins
    ):
        raise ValueError("reference_histograms has the wrong shape or too few samples.")
    if not isinstance(truth, torch.Tensor) or tuple(truth.shape) != (
        groups,
        spec.num_spectrum_bins,
    ):
        raise ValueError("truth_histograms has the wrong shape.")

    for label, artifact in model_artifacts.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("every model must have a nonempty label.")
        if not isinstance(artifact, dict):
            raise ValueError(f"model artifact {label!r} must be a dictionary.")
        if artifact.get("protocol_id") != spec.protocol_id:
            raise ValueError(f"model {label!r} uses a different protocol_id.")
        if artifact.get("specification_hash") != spec.specification_hash:
            raise ValueError(f"model {label!r} uses a different specification hash.")
        model_ids = artifact.get("condition_ids")
        model_measurements = artifact.get("measurements")
        samples = artifact.get("samples")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (model_ids, model_measurements, samples)
        ):
            raise ValueError(f"model {label!r} is missing required tensors.")
        if not _same_tensor(condition_ids, model_ids, atol=0.0):
            raise ValueError(
                f"model {label!r} condition IDs differ from the reference."
            )
        if not _same_tensor(measurements, model_measurements, atol=1e-7):
            raise ValueError(f"model {label!r} measurements differ from the reference.")
        if (
            samples.shape[0] != groups
            or samples.shape[1] < qualitative_sample_count
            or samples.shape[2] != spec.num_spectrum_bins
        ):
            raise ValueError(f"model {label!r} samples have the wrong shape or K<16.")
        _validate_simplex(samples, name=label)
        validate_optional_forward_spectra(
            artifact,
            groups=groups,
            num_bins=spec.num_spectrum_bins,
            artifact_name=f"model {label!r}",
        )
        model_grid = artifact.get("grid")
        if not isinstance(model_grid, dict):
            raise ValueError(f"model {label!r} is missing its grid.")
        if model_grid.get("specification_hash") != spec.specification_hash:
            raise ValueError(f"model {label!r} grid hash differs.")
        for field in _GRID_FIELDS:
            if field not in model_grid or not isinstance(
                model_grid[field], torch.Tensor
            ):
                raise ValueError(f"model {label!r} grid is missing {field!r}.")
            expected = reference_grid[field]
            tolerance = 0.0 if not expected.is_floating_point() else 1e-12
            if not _same_tensor(model_grid[field], expected, atol=tolerance):
                raise ValueError(f"model {label!r} grid field {field!r} differs.")
    return spec


def _safe_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("._-")
    return slug or "model"


def _model_color(label: str) -> Any:
    """Keep method colors fixed when comparison groups or NFE change."""

    key = re.sub(r"[^a-z0-9]", "", label.lower())
    if key.startswith(("qwfm", "quantilewasserstein")):
        return next(
            (
                color
                for size, color in (
                    ("512", "#7570B3"),
                    ("256", "#D95F02"),
                    ("128", "#1B9E77"),
                )
                if size in key
            ),
            "#1B9E77",
        )
    if key.startswith(("hfm", "histogramlinear")):
        return "#222222"
    if key.startswith(("regression", "deterministicregression")):
        return "#56B4E9" if "w1" in key else "#377EB8"
    if key.startswith(("cvae", "conditionalvae")):
        return "#AD1457" if "z32" in key else "#E7298A"
    if key.startswith(("cnf", "conditionalnormalizing")):
        return "#E41A1C"
    if "nnls" in key:
        return "#7F7F7F"
    if "x2" in key:
        return "#A6761D"
    if "discrepancy" in key:
        return "#8C564B"
    if "bayes" in key:
        return "#B15928" if "official" in key else "#BCBD22"
    index = int.from_bytes(hashlib.sha256(label.encode()).digest()[:2], "little") % 10
    return plt.get_cmap("tab10")(index)


def _save_figure(figure: Any, path: Path, *, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _spectrum_tile(
    figure: Any,
    position: Any,
    samples: torch.Tensor,
    rate_centers: torch.Tensor,
    *,
    title: str,
    color: Any,
    maximum_mass: float,
    bottom: bool = True,
    left: bool = True,
) -> Any:
    axis = figure.add_subplot(position)
    rates = rate_centers.cpu().numpy()
    axis.plot(
        rates, samples.cpu().numpy(), color=color, linewidth=1.2, drawstyle="steps-mid"
    )
    axis.set(
        xscale="log", xlim=(float(rates[0]), float(rates[-1])), ylim=(0.0, maximum_mass)
    )
    axis.set_title(title, fontsize=8, pad=3)
    axis.tick_params(labelsize=7, labelbottom=bottom, labelleft=left)
    if bottom:
        axis.set_xlabel(r"$R$ (s$^{-1}$)", fontsize=8)
    if left:
        axis.set_ylabel("Bin mass", fontsize=8)
    axis.grid(alpha=0.18, linewidth=0.5)
    return axis


def _add_sample_tiles(
    figure: Any,
    position: Any,
    samples: torch.Tensor,
    rate_centers: torch.Tensor,
    *,
    label: str,
    color: Any,
    maximum_mass: float,
    rows: int,
    columns: int,
) -> None:
    if samples.shape[0] == 1:
        layout = position.subgridspec(1, 4)
        _spectrum_tile(
            figure,
            layout[0, 1:3],
            samples[0],
            rate_centers,
            title=f"{label}: unique prediction (repeated for K shots)",
            color=color,
            maximum_mass=maximum_mass,
        )
        return
    if samples.shape[0] != rows * columns:
        raise ValueError("Each stochastic gallery must display exactly K=16 shots.")
    layout = position.subgridspec(rows, columns, hspace=0.46, wspace=0.28)
    bounds = position.get_position(figure)
    figure.text(bounds.x0, bounds.y1 + 0.022, label, fontsize=10, fontweight="bold")
    for index, sample in enumerate(samples):
        row, column = divmod(index, columns)
        title = f"Shot {index + 1:02d}"
        _spectrum_tile(
            figure,
            layout[row, column],
            sample,
            rate_centers,
            title=title,
            color=color,
            maximum_mass=maximum_mass,
            bottom=row == rows - 1,
            left=column == 0,
        )


def _plot_kshot_gallery(
    reference_samples: torch.Tensor,
    models: dict[str, torch.Tensor],
    truth: torch.Tensor,
    measurement: torch.Tensor,
    rate_centers: torch.Tensor,
    echo_times: torch.Tensor,
    *,
    condition_id: int,
    path: Path,
    maximum_mass: float,
    dpi: int,
    combined: bool,
    nfe: int | None = None,
) -> None:
    """Only observed condition, reference spectra, and resolved K-shot curves."""

    rows, columns = (2, 8) if combined else (4, 4)
    method_heights = [
        1.7 if samples.shape[0] == 1 else (3.6 if combined else 7.2)
        for samples in models.values()
    ]
    figure = plt.figure(figsize=(20 if combined else 15, 6.3 + sum(method_heights)))
    try:
        outer = figure.add_gridspec(
            2 + len(models),
            1,
            height_ratios=[1.8, 3.5, *method_heights],
            left=0.048,
            right=0.99,
            bottom=0.035,
            top=0.952,
            hspace=0.42,
        )
        header = outer[0].subgridspec(1, 4, wspace=0.5)
        condition_axis = figure.add_subplot(header[0, :2])
        condition_axis.plot(
            echo_times.cpu().numpy() * 1000,
            measurement.cpu().numpy(),
            color="#111111",
            marker="o",
            markersize=2,
            linewidth=1.2,
        )
        condition_axis.set(
            xscale="log",
            title="Observed condition",
            xlabel="Echo time (ms)",
            ylabel="Normalized signal",
        )
        condition_axis.tick_params(labelsize=8)
        condition_axis.grid(alpha=0.18)
        _spectrum_tile(
            figure,
            header[0, 2:],
            truth,
            rate_centers,
            title="Simulator truth (one realized spectrum)",
            color="#0072B2",
            maximum_mass=maximum_mass,
        )
        _add_sample_tiles(
            figure,
            outer[1],
            reference_samples,
            rate_centers,
            label="SMC posterior reference: 16 samples",
            color="#0072B2",
            maximum_mass=maximum_mass,
            rows=2,
            columns=8,
        )
        for row, (label, samples) in enumerate(models.items(), start=2):
            _add_sample_tiles(
                figure,
                outer[row],
                samples,
                rate_centers,
                label=label,
                color=_model_color(label),
                maximum_mass=maximum_mass,
                rows=rows,
                columns=columns,
            )
        title = "All methods" if combined else next(iter(models))
        nfe_text = "" if nfe is None else f" | FM Euler NFE={nfe}"
        figure.suptitle(
            f"Condition {condition_id} | {title} | K=16{nfe_text}",
            fontsize=14,
            fontweight="bold",
        )
        _save_figure(figure, path, dpi=dpi)
    finally:
        plt.close(figure)


def _plot_combined_bands(
    reference: torch.Tensor,
    models: dict[str, torch.Tensor],
    truth: torch.Tensor,
    rate_centers: torch.Tensor,
    *,
    condition_id: int,
    path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.8))
    rates = rate_centers.cpu().numpy()
    collections = [("SMC reference", reference), *models.items()]
    colors = ["#111827", *(_model_color(label) for label in models)]
    for (label, samples), color in zip(collections, colors):
        quantiles = torch.quantile(
            samples.to(torch.float64),
            torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64),
            dim=0,
        )
        axis.fill_between(
            rates,
            quantiles[0].cpu().numpy(),
            quantiles[2].cpu().numpy(),
            color=color,
            alpha=0.08 if label != "SMC reference" else 0.16,
            linewidth=0,
        )
        axis.plot(
            rates,
            quantiles[1].cpu().numpy(),
            color=color,
            linewidth=2.2 if label == "SMC reference" else 1.4,
            label=f"{label}: median / 5--95%",
        )
    axis.plot(
        rates,
        truth.cpu().numpy(),
        color="#dc2626",
        linestyle="--",
        linewidth=1.7,
        label="simulator truth (calibration only)",
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"relaxation rate $R$ (s$^{-1}$)")
    axis.set_ylabel("histogram bin mass")
    axis.set_title(f"Condition {condition_id}: conditional posterior summaries")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _save_figure(figure, path, dpi=dpi)


def _plot_forward_residuals(
    reference: torch.Tensor,
    models: dict[str, torch.Tensor],
    measurement: torch.Tensor,
    echo_times: torch.Tensor,
    kernel: torch.Tensor,
    *,
    sigma: float,
    condition_id: int,
    path: Path,
    dpi: int,
) -> None:
    collections = [("SMC reference", reference), *models.items()]
    colors = ["#111827", *(_model_color(label) for label in models)]
    figure, axes = plt.subplots(
        len(collections),
        2,
        figsize=(13, max(3.0, 2.4 * len(collections))),
        squeeze=False,
        sharex="col",
    )
    echoes = echo_times.cpu().numpy()
    observed = measurement.to(torch.float64)
    for row, ((label, samples), color) in enumerate(zip(collections, colors)):
        prediction = samples.to(torch.float64) @ kernel.to(torch.float64).T
        predicted_quantiles = torch.quantile(
            prediction,
            torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64),
            dim=0,
        )
        residual_quantiles = (predicted_quantiles - observed[None, :]) / sigma
        fit_axis, residual_axis = axes[row]
        fit_axis.scatter(
            echoes, observed.cpu().numpy(), s=13, color="#dc2626", label="y"
        )
        fit_axis.fill_between(
            echoes,
            predicted_quantiles[0].cpu().numpy(),
            predicted_quantiles[2].cpu().numpy(),
            color=color,
            alpha=0.18,
        )
        fit_axis.plot(
            echoes, predicted_quantiles[1].cpu().numpy(), color=color, linewidth=1.5
        )
        residual_axis.fill_between(
            echoes,
            residual_quantiles[0].cpu().numpy(),
            residual_quantiles[2].cpu().numpy(),
            color=color,
            alpha=0.18,
        )
        residual_axis.plot(
            echoes, residual_quantiles[1].cpu().numpy(), color=color, linewidth=1.5
        )
        residual_axis.axhline(0.0, color="#6b7280", linewidth=0.8)
        fit_axis.set_ylabel(label, fontsize=9)
        for axis in (fit_axis, residual_axis):
            axis.set_xscale("log")
            axis.grid(alpha=0.18)
    axes[0, 0].set_title("Observed decay and posterior forward prediction")
    axes[0, 1].set_title(r"Prediction residual $(Ah-y)/\sigma$")
    axes[-1, 0].set_xlabel(r"sampling delay $\tau$ (s)")
    axes[-1, 1].set_xlabel(r"sampling delay $\tau$ (s)")
    figure.suptitle(f"Condition {condition_id}: forward consistency", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(figure, path, dpi=dpi)


def _plot_downstream_quantities(
    reference: torch.Tensor,
    models: dict[str, torch.Tensor],
    log_rate_edges: torch.Tensor,
    *,
    condition_id: int,
    path: Path,
    dpi: int,
) -> None:
    collections = [("SMC reference", reference), *models.items()]
    derived = [
        derived_quantities(samples, log_rate_edges)
        for label, samples in collections
    ]
    quantity_names = tuple(derived[0].keys())
    rows = (len(quantity_names) + 2) // 3
    figure, axes = plt.subplots(rows, 3, figsize=(16, 4.5 * rows), squeeze=False)
    labels = [label for label, _ in collections]
    colors = ["#374151", *(_model_color(label) for label in models)]
    display_names = {
        "rate_2_10": r"mass fraction: $2\leq R<10$",
        "rate_10_50": r"mass fraction: $10\leq R<50$",
        "rate_50_250": r"mass fraction: $50\leq R\leq250$",
        "mean_log_rate": r"mean $\log R$",
        "mean_rate": r"mean rate $R$ (s$^{-1}$)",
        "effective_peak_count": "effective peak count (5% threshold)",
    }
    for axis, name in zip(axes.flat, quantity_names):
        values = [item[name].cpu().numpy() for item in derived]
        boxplot = axis.boxplot(
            values,
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
            widths=0.65,
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.42)
        axis.set_title(display_names.get(name, name))
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(quantity_names) :]:
        axis.axis("off")
    figure.suptitle(
        f"Condition {condition_id}: downstream posterior quantities", fontsize=14
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(figure, path, dpi=dpi)


def visualize_posterior_artifacts(
    reference: dict[str, Any],
    model_artifacts: dict[str, dict[str, Any]],
    output_dir: str | Path,
    *,
    condition_indices: list[int] | None = None,
    qualitative_sample_count: int = 16,
    dpi: int = 180,
    overwrite: bool = False,
    include_diagnostics: bool = False,
    nfe: int | None = None,
    maximum_mass_by_condition: dict[int, float] | None = None,
    reused_model_sheets: dict[int, dict[str, str]] | None = None,
    model_sheet_directories: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Save combined and per-method galleries; diagnostics are opt-in only.

    An NFE sweep supplies shared axis limits and previously generated static
    model sheets. Stored samples are never reordered, smoothed, or regenerated.
    """

    spec = validate_visualization_artifacts(
        reference,
        model_artifacts,
        qualitative_sample_count=qualitative_sample_count,
    )
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi must be an integer of at least 72.")
    groups = int(reference["condition_ids"].shape[0])
    selected = (
        list(range(groups)) if condition_indices is None else list(condition_indices)
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("condition_indices must be nonempty and unique.")
    if any(index < 0 or index >= groups for index in selected):
        raise ValueError("condition_indices contains an out-of-range value.")
    slugs = {label: _safe_label(label) for label in model_artifacts}
    if len(set(slugs.values())) != len(slugs):
        raise ValueError("model labels collide after filename sanitization.")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    grid = reference["grid"]
    rate_centers = grid["rate_centers"].to(device="cpu", dtype=torch.float64)
    log_rate_edges = grid["log_rate_edges"].to(device="cpu", dtype=torch.float64)
    echo_times = grid["echo_times"].to(device="cpu", dtype=torch.float64)
    kernel = grid["kernel"].to(device="cpu", dtype=torch.float64)
    manifest_conditions: list[dict[str, Any]] = []

    for condition_index in selected:
        condition_id = int(reference["condition_ids"][condition_index])
        condition_dir = destination / f"condition_{condition_index:04d}"
        condition_dir.mkdir(parents=True, exist_ok=True)
        reference_samples = reference["reference_histograms"][condition_index].to(
            device="cpu", dtype=torch.float64
        )
        model_samples = {
            label: artifact["samples"][condition_index].to(
                device="cpu", dtype=torch.float64
            )
            for label, artifact in model_artifacts.items()
        }
        reused = (reused_model_sheets or {}).get(condition_index, {})
        model_paths = {
            label: Path(reused[label])
            if label in reused
            else (model_sheet_directories or {}).get(label, destination)
            / f"condition_{condition_index:04d}"
            / f"{slugs[label]}_k16.png"
            for label in model_samples
        }
        for label in reused:
            if label in model_paths and not model_paths[label].is_file():
                raise FileNotFoundError(
                    f"Missing reused model sheet: {model_paths[label]}"
                )
        combined_path = condition_dir / "combined_k16.png"
        expected_files = [
            combined_path,
            *(path for label, path in model_paths.items() if label not in reused),
        ]
        diagnostic_paths = (
            {
                "combined_posterior_band": condition_dir
                / "diagnostics/combined_posterior_band.png",
                "forward_residual": condition_dir / "diagnostics/forward_residual.png",
                "downstream_quantities": condition_dir
                / "diagnostics/downstream_quantities.png",
            }
            if include_diagnostics
            else {}
        )
        expected_files.extend(diagnostic_paths.values())
        if not overwrite and any(path.exists() for path in expected_files):
            raise FileExistsError(
                f"qualitative output already exists in {condition_dir}; pass --overwrite."
            )
        truth = reference["truth_histograms"][condition_index].to(
            device="cpu", dtype=torch.float64
        )
        measurement = reference["measurements"][condition_index].to(
            device="cpu", dtype=torch.float64
        )
        maximum_mass = max(
            float(truth.max().item()),
            float(reference_samples[:qualitative_sample_count].max().item()),
            *(
                float(samples[:qualitative_sample_count].max().item())
                for samples in model_samples.values()
            ),
        )
        maximum_mass = max(1.0e-6, 1.05 * maximum_mass)
        if maximum_mass_by_condition is not None:
            shared = maximum_mass_by_condition[condition_index]
            if not 0 < shared or shared < maximum_mass - 1e-12:
                raise ValueError(
                    "Shared spectrum axis limit would clip displayed samples."
                )
            maximum_mass = shared
        display_models = {}
        for label, samples in model_samples.items():
            deterministic = bool(model_artifacts[label].get("deterministic", False))
            if deterministic and not torch.equal(
                samples, samples[:1].expand_as(samples)
            ):
                raise ValueError(
                    f"Deterministic model {label!r} stores varying predictions."
                )
            display_models[label] = samples[
                : 1 if deterministic else qualitative_sample_count
            ]
        gallery_arguments = dict(
            reference_samples=reference_samples[:qualitative_sample_count],
            truth=truth,
            measurement=measurement,
            rate_centers=rate_centers,
            echo_times=echo_times,
            condition_id=condition_id,
            maximum_mass=maximum_mass,
            dpi=dpi,
        )
        _plot_kshot_gallery(
            models=display_models,
            path=combined_path,
            combined=True,
            nfe=nfe,
            **gallery_arguments,
        )
        for label, samples in display_models.items():
            if label in reused:
                continue
            model_paths[label].parent.mkdir(parents=True, exist_ok=True)
            model_nfe = (
                nfe
                if model_artifacts[label].get("method")
                in ("histogram_linear", "quantile_wasserstein")
                else None
            )
            _plot_kshot_gallery(
                models={label: samples},
                path=model_paths[label],
                combined=False,
                nfe=model_nfe,
                **gallery_arguments,
            )
        if include_diagnostics:
            (condition_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
            model_forward_samples = {
                label: artifact.get("forward_spectra", artifact["samples"])[
                    condition_index
                ].to(device="cpu", dtype=torch.float64)
                for label, artifact in model_artifacts.items()
            }
            _plot_combined_bands(
                reference_samples,
                model_samples,
                truth,
                rate_centers,
                condition_id=condition_id,
                path=diagnostic_paths["combined_posterior_band"],
                dpi=dpi,
            )
            _plot_forward_residuals(
                reference_samples,
                model_forward_samples,
                measurement,
                echo_times,
                kernel,
                sigma=spec.noise_standard_deviation,
                condition_id=condition_id,
                path=diagnostic_paths["forward_residual"],
                dpi=dpi,
            )
            _plot_downstream_quantities(
                reference_samples,
                model_samples,
                log_rate_edges,
                condition_id=condition_id,
                path=diagnostic_paths["downstream_quantities"],
                dpi=dpi,
            )
        manifest_conditions.append(
            {
                "condition_index": condition_index,
                "condition_id": condition_id,
                "combined_k16": str(combined_path),
                "model_sheets": [str(path) for path in model_paths.values()],
                "model_sheets_by_label": {
                    label: str(path) for label, path in model_paths.items()
                },
                "displayed_shots_by_model": {
                    label: int(samples.shape[0])
                    for label, samples in display_models.items()
                },
                "reference_sample_indices": list(range(qualitative_sample_count)),
                "spectrum_ymax": maximum_mass,
                "diagnostics": {
                    key: str(path) for key, path in diagnostic_paths.items()
                },
            }
        )

    manifest = {
        "protocol_version": VISUALIZATION_PROTOCOL_VERSION,
        "specification_hash": spec.specification_hash,
        "qualitative_sample_count": qualitative_sample_count,
        "include_diagnostics": include_diagnostics,
        "nfe": nfe,
        "sample_selection": "saved K-prefix; no sorting, smoothing, or sample selection by quality",
        "reference_role": "SMC posterior draws, distinct from the simulator truth",
        "model_labels": list(model_artifacts),
        "representation_usage": {
            "spectrum_sheets_and_posterior_bands": (
                NORMALIZED_HISTOGRAM_REPRESENTATION
            ),
            "forward_residual_by_model": {
                label: (
                    RAW_FORWARD_SPECTRUM_REPRESENTATION
                    if artifact.get("forward_spectra") is not None
                    else NORMALIZED_HISTOGRAM_REPRESENTATION
                )
                for label, artifact in model_artifacts.items()
            },
        },
        "conditions": manifest_conditions,
    }
    atomic_write_json(destination / "visualization_manifest.json", manifest)
    return manifest


def _parse_model_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"model argument {value!r} must have LABEL=ARTIFACT.pt form."
            )
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in paths:
            raise ValueError("model labels must be nonempty and unique.")
        path = Path(raw_path).expanduser().resolve()
        paths[label] = path
    return paths


def _parse_model_arguments(
    values: list[str],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = _parse_model_paths(values)
    return paths, {label: load_torch_artifact(path) for label, path in paths.items()}


def _path_at_nfe(path: Path, nfe: int) -> Path:
    """Change only an exact nfe_XXXX directory component, never model names."""

    positions = [
        i for i, part in enumerate(path.parts) if re.fullmatch(r"nfe_\d+", part)
    ]
    if len(positions) > 1:
        raise ValueError(f"Ambiguous NFE directory in model path: {path}")
    parts = list(path.parts)
    if positions:
        parts[positions[0]] = f"nfe_{nfe:04d}"
    return Path(*parts)


def _shared_spectrum_limits(
    reference: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[int, float]:
    maximum = (
        reference["reference_histograms"][:, :16]
        .detach()
        .cpu()
        .double()
        .amax(dim=(-1, -2))
    )
    maximum = torch.maximum(
        maximum, reference["truth_histograms"].detach().cpu().double().amax(dim=-1)
    )
    for artifact in artifacts:
        maximum = torch.maximum(
            maximum,
            artifact["samples"][:, :16].detach().cpu().double().amax(dim=(-1, -2)),
        )
    return {
        index: max(1e-6, 1.05 * value) for index, value in enumerate(maximum.tolist())
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cached conditional posterior sample artifacts."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="LABEL=ARTIFACT.pt",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_indices", type=int, nargs="+", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include_diagnostics",
        action="store_true",
        help="Also save posterior bands, residuals, and derived-quantity plots.",
    )
    parser.add_argument(
        "--nfe_values",
        type=int,
        nargs="+",
        default=None,
        help="Replace FM nfe_XXXX directories; static methods are reused.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Warn and record absent model files; invalid artifacts still fail.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reference_path = Path(args.reference).expanduser().resolve()
    reference = load_torch_artifact(reference_path)
    source_paths = _parse_model_paths(args.models)
    values = args.nfe_values if args.nfe_values is not None else [None]
    if args.nfe_values is not None and (
        not values
        or len(set(values)) != len(values)
        or any(value <= 0 for value in values)
    ):
        raise ValueError("nfe_values must be nonempty, positive, and unique.")
    cache: dict[Path, dict[str, Any]] = {}
    jobs = []
    skipped_sources = []
    for nfe in values:
        paths, artifacts = {}, {}
        for label, source in source_paths.items():
            path = source if nfe is None else _path_at_nfe(source, nfe)
            if not path.is_file():
                if not args.skip_missing or path.exists():
                    raise FileNotFoundError(path)
                skipped_sources.append({"label": label, "path": str(path), "nfe": nfe})
                warnings.warn(f"Skipping missing model {label!r}: {path}", stacklevel=2)
                continue
            if path not in cache:
                cache[path] = load_torch_artifact(path)
            paths[label], artifacts[label] = path, cache[path]
        # Validate every available NFE before creating any image.
        validate_visualization_artifacts(reference, artifacts)
        jobs.append((nfe, paths, artifacts))
    shared_limits = _shared_spectrum_limits(reference, list(cache.values()))
    destination = Path(args.output_dir).resolve()
    reused: dict[int, dict[str, str]] = {}
    manifests = []
    static_labels = {
        label
        for label, path in source_paths.items()
        if not any(re.fullmatch(r"nfe_\d+", part) for part in path.parts)
    }
    for nfe, paths, artifacts in jobs:
        output = destination if nfe is None else destination / f"nfe_{nfe:04d}"
        static_directories = (
            {
                label: destination / "nfe_independent"
                for label in artifacts
                if label in static_labels
            }
            if nfe is not None
            else {}
        )
        manifest = visualize_posterior_artifacts(
            reference,
            artifacts,
            output,
            condition_indices=args.condition_indices,
            dpi=args.dpi,
            overwrite=args.overwrite,
            include_diagnostics=args.include_diagnostics,
            nfe=nfe,
            maximum_mass_by_condition=shared_limits,
            reused_model_sheets=reused,
            model_sheet_directories=static_directories,
        )
        manifest["source_paths"] = {
            "reference": str(reference_path),
            "models": {label: str(path) for label, path in paths.items()},
        }
        manifest["skipped_sources"] = [
            item for item in skipped_sources if item["nfe"] == nfe
        ]
        atomic_write_json(output / "visualization_manifest.json", manifest)
        if nfe is not None:
            for condition in manifest["conditions"]:
                reused[condition["condition_index"]] = {
                    label: path
                    for label, path in condition["model_sheets_by_label"].items()
                    if label in static_labels
                }
        manifests.append(manifest)
        print(
            f"saved galleries for {len(manifest['conditions'])} conditions"
            + ("" if nfe is None else f" at NFE={nfe}"),
            flush=True,
        )
    root_manifest = (
        manifests[0]
        if args.nfe_values is None
        else {
            "protocol_version": VISUALIZATION_PROTOCOL_VERSION,
            "nfe_values": values,
            "static_method_sheets": "nfe_independent/condition_XXXX",
            "shared_spectrum_y_limits_across_nfe": True,
            "skipped_sources": skipped_sources,
            "runs": manifests,
        }
    )
    atomic_write_json(destination / "visualization_manifest.json", root_manifest)


if __name__ == "__main__":
    main()
