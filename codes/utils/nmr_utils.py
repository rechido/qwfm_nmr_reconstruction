from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import torch


def make_condition(
    measurements: torch.Tensor,
    echo_times: torch.Tensor,
    output_size: int,
    noise_standard_deviation: float,
) -> torch.Tensor:
    """Broadcast the complete y/a vectors and sigma to every state position."""

    if measurements.ndim != 2:
        raise ValueError("measurements must have shape [batch, measurements].")
    if echo_times.ndim != 1 or echo_times.shape[0] != measurements.shape[1]:
        raise ValueError(
            "echo_times must be one-dimensional and match the measurement count."
        )
    if (
        isinstance(output_size, bool)
        or not isinstance(output_size, int)
        or output_size < 2
    ):
        raise ValueError("output_size must be an integer of at least two.")
    if (
        isinstance(noise_standard_deviation, bool)
        or not isinstance(noise_standard_deviation, (int, float))
        or not math.isfinite(float(noise_standard_deviation))
        or noise_standard_deviation <= 0.0
    ):
        raise ValueError("noise_standard_deviation must be finite and positive.")
    if (
        not bool(torch.isfinite(echo_times).all())
        or bool((echo_times <= 0).any())
        or bool((torch.diff(echo_times) <= 0).any())
    ):
        raise ValueError(
            "echo_times must be finite, positive, and strictly increasing."
        )
    if not bool(torch.isfinite(measurements).all()):
        raise ValueError("measurements must contain only finite values.")

    state_length = int(output_size)
    measurement_channels = measurements[:, :, None].expand(
        -1,
        -1,
        state_length,
    )

    log_echo = echo_times.to(device=measurements.device, dtype=measurements.dtype).log()
    normalized_echo = (log_echo - log_echo[0]) / (log_echo[-1] - log_echo[0])
    echo_channels = normalized_echo[None, :, None].expand(
        measurements.shape[0],
        -1,
        state_length,
    )
    sigma_channel = torch.full(
        (measurements.shape[0], 1, state_length),
        float(noise_standard_deviation),
        device=measurements.device,
        dtype=measurements.dtype,
    )
    return torch.cat((measurement_channels, echo_channels, sigma_channel), dim=1)


def sample_standard_dirichlet(
    batch_size: int,
    num_bins: int,
    device: torch.device,
    output_dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw Dirichlet(1_N) in float64 without a numerical floor.

    Gamma(1, 1) is exponential, so -log(U) followed by normalization is
    exactly the Gamma construction in the proposal. Invalid rows are redrawn.
    """

    result = torch.empty(batch_size, num_bins, device=device, dtype=torch.float64)
    remaining = torch.arange(batch_size, device=device)
    while int(remaining.numel()) > 0:
        uniforms = torch.rand(
            remaining.numel(),
            num_bins,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        gamma_draws = -torch.log(uniforms)
        totals = gamma_draws.sum(dim=1, keepdim=True)
        valid = torch.isfinite(gamma_draws).all(dim=1) & (gamma_draws > 0).all(dim=1)
        valid = valid & torch.isfinite(totals[:, 0]) & (totals[:, 0] > 0)
        if bool(valid.any()):
            result[remaining[valid]] = gamma_draws[valid] / totals[valid]
        remaining = remaining[~valid]
    return result.to(dtype=output_dtype)


def histogram_to_quantiles(
    histograms: torch.Tensor,
    log_rate_edges: torch.Tensor,
    num_quantiles: Optional[int] = None,
) -> torch.Tensor:
    """Invert a piecewise-constant histogram CDF at midpoint probabilities."""

    if histograms.ndim != 2:
        raise ValueError("histograms must have shape [batch, bins].")
    if log_rate_edges.numel() != histograms.shape[1] + 1:
        raise ValueError("log_rate_edges must have one more entry than the histogram.")

    output_dtype = histograms.dtype
    work = histograms.to(torch.float64)
    work = work / work.sum(dim=1, keepdim=True)
    edges = log_rate_edges.to(device=work.device, dtype=torch.float64)
    quantile_count = int(num_quantiles or histograms.shape[1])
    probabilities = (
        torch.arange(quantile_count, device=work.device, dtype=torch.float64) + 0.5
    ) / float(quantile_count)
    probabilities = probabilities[None, :].expand(work.shape[0], -1).contiguous()

    cdf = torch.cat(
        (
            torch.zeros(work.shape[0], 1, device=work.device, dtype=torch.float64),
            work.cumsum(dim=1),
        ),
        dim=1,
    )
    cdf[:, -1] = 1.0
    bin_indices = torch.searchsorted(cdf.contiguous(), probabilities, right=True) - 1
    bin_indices = bin_indices.clamp(0, work.shape[1] - 1)

    left_cdf = cdf.gather(1, bin_indices)
    selected_mass = work.gather(1, bin_indices)
    if bool((selected_mass <= 0).any()) or not bool(
        torch.isfinite(selected_mass).all()
    ):
        raise RuntimeError("A midpoint probability selected a zero-mass histogram bin.")

    left_edges = edges[bin_indices]
    right_edges = edges[bin_indices + 1]
    within_bin = ((probabilities - left_cdf) / selected_mass).clamp(0.0, 1.0)
    quantiles = left_edges + within_bin * (right_edges - left_edges)
    return quantiles.to(output_dtype)


def make_normalized_log_rate_edges(log_rate_edges: torch.Tensor) -> torch.Tensor:
    """Affine-normalize a strictly increasing log-rate grid onto ``[0, 1]``.

    Quantile-Wasserstein flow states use this dimensionless coordinate so their
    input and velocity scales do not depend on the physical rate unit or the
    width of the configured log-rate interval.  The returned tensor preserves
    the input dtype and device.
    """

    if not isinstance(log_rate_edges, torch.Tensor):
        raise TypeError("log_rate_edges must be a torch.Tensor.")
    if log_rate_edges.ndim != 1 or log_rate_edges.numel() < 3:
        raise ValueError(
            "log_rate_edges must be one-dimensional and contain at least three edges."
        )
    if not log_rate_edges.is_floating_point():
        raise TypeError("log_rate_edges must have a floating-point dtype.")
    if not bool(torch.isfinite(log_rate_edges).all()) or bool(
        (torch.diff(log_rate_edges) <= 0.0).any()
    ):
        raise ValueError("log_rate_edges must be finite and strictly increasing.")

    normalized = (log_rate_edges - log_rate_edges[0]) / (
        log_rate_edges[-1] - log_rate_edges[0]
    )
    # Pin the endpoints exactly.  This makes clipping and boundary assignment
    # deterministic even for low-precision state tensors.
    normalized = normalized.clone()
    normalized[0] = 0.0
    normalized[-1] = 1.0
    return normalized.contiguous()


def quantile_wasserstein_representation_signature() -> dict[str, str]:
    """Return the immutable coordinate/decoder contract stored in checkpoints."""

    return {
        "protocol_version": "normalized_log_quantile_hard_bin_v2",
        "state_coordinate": "affine_normalized_log_rate_[0,1]",
        "quantile_probability_grid": "midpoints_(index+0.5)/num_quantiles",
        "histogram_decoder": "equal_mass_bin_edge_hard_assignment",
        "internal_edge_policy": "right_bin_and_upper_endpoint_to_last_bin",
    }


def quantiles_to_histograms(
    quantiles: torch.Tensor,
    log_rate_edges: torch.Tensor,
) -> torch.Tensor:
    """Decode equal-mass particles into histogram bin masses.

    Each particle contributes exactly ``1 / L`` to the bin interval containing
    it.  Internal edges belong to the bin on their right, while the global upper
    endpoint belongs to the final bin.  Unlike the previous center-based linear
    splat, this is an asymptotically consistent inverse of midpoint inverse-CDF
    sampling: every bin-mass round-trip error is bounded by ``1 / L``.
    """

    if not isinstance(quantiles, torch.Tensor):
        raise TypeError("quantiles must be a torch.Tensor.")
    if quantiles.ndim != 2:
        raise ValueError("quantiles must have shape [batch, particles].")
    if quantiles.shape[1] < 1:
        raise ValueError("quantiles must contain at least one particle per batch row.")
    if not quantiles.is_floating_point():
        raise TypeError("quantiles must have a floating-point dtype.")
    if not isinstance(log_rate_edges, torch.Tensor):
        raise TypeError("log_rate_edges must be a torch.Tensor.")
    if log_rate_edges.ndim != 1 or log_rate_edges.numel() < 3:
        raise ValueError(
            "log_rate_edges must be one-dimensional and contain at least three edges."
        )
    if not log_rate_edges.is_floating_point():
        raise TypeError("log_rate_edges must have a floating-point dtype.")

    edges = log_rate_edges.to(
        device=quantiles.device,
        dtype=quantiles.dtype,
    ).contiguous()
    if not bool(torch.isfinite(edges).all()) or bool((torch.diff(edges) <= 0.0).any()):
        raise ValueError("log_rate_edges must be finite and strictly increasing.")
    if not bool(torch.isfinite(quantiles).all()):
        raise ValueError("quantiles must contain only finite values.")

    if bool((quantiles < edges[0]).any()) or bool(
        (quantiles > edges[-1]).any()
    ):
        raise ValueError("quantiles must lie inside the supplied edge domain.")

    num_bins = int(edges.numel() - 1)
    bin_indices = torch.searchsorted(
        edges,
        quantiles.contiguous(),
        right=True,
    ) - 1
    bin_indices = bin_indices.clamp_(0, num_bins - 1)
    histograms = torch.zeros(
        quantiles.shape[0],
        num_bins,
        device=quantiles.device,
        dtype=quantiles.dtype,
    )
    particle_mass = 1.0 / float(quantiles.shape[1])
    histograms.scatter_add_(
        1,
        bin_indices,
        torch.full_like(quantiles, particle_mass),
    )
    return histograms


def histogram_w1(
    first: torch.Tensor,
    second: torch.Tensor,
    log_rate_edges: torch.Tensor,
) -> torch.Tensor:
    """Exact 1D W1 for piecewise-uniform density inside each log-rate bin."""

    widths = torch.diff(log_rate_edges.to(device=first.device, dtype=first.dtype))
    mass_difference = first - second
    boundary_difference = torch.cat(
        (
            torch.zeros_like(mass_difference[..., :1]),
            mass_difference.cumsum(dim=-1),
        ),
        dim=-1,
    )
    left = boundary_difference[..., :-1]
    right = boundary_difference[..., 1:]
    same_sign_area = 0.5 * widths * (left.abs() + right.abs())
    crossing_area = 0.5 * widths * (left.square() + right.square())
    crossing_area = crossing_area / (left.abs() + right.abs()).clamp_min(1.0e-15)
    bin_area = torch.where(left * right >= 0.0, same_sign_area, crossing_area)
    return bin_area.sum(dim=-1)


def histogram_w2(
    first: torch.Tensor,
    second: torch.Tensor,
    log_rate_edges: torch.Tensor,
    w2_num_quantiles: int,
) -> torch.Tensor:
    """Midpoint-quadrature 1D W2 on the natural-log relaxation-rate axis."""

    if first.shape != second.shape:
        raise ValueError("The two histogram tensors must have identical shapes.")
    if (
        isinstance(w2_num_quantiles, bool)
        or not isinstance(w2_num_quantiles, int)
        or w2_num_quantiles < 1
    ):
        raise ValueError("w2_num_quantiles must be a positive integer.")
    leading_shape = first.shape[:-1]
    first_quantiles = histogram_to_quantiles(
        first.reshape(-1, first.shape[-1]),
        log_rate_edges,
        num_quantiles=w2_num_quantiles,
    )
    second_quantiles = histogram_to_quantiles(
        second.reshape(-1, second.shape[-1]),
        log_rate_edges,
        num_quantiles=w2_num_quantiles,
    )
    distances = (first_quantiles - second_quantiles).square().mean(dim=-1).sqrt()
    return distances.reshape(leading_shape)


def evaluate_histogram_samples(
    samples: torch.Tensor,
    targets: torch.Tensor,
    measurements: torch.Tensor,
    kernel: torch.Tensor,
    log_rate_edges: torch.Tensor,
    noise_standard_deviation: float,
    w2_num_quantiles: int,
    peak_threshold_fraction: float,
) -> Dict[str, float | str]:
    """Metrics shared by the two decoders on a common histogram grid."""

    if samples.ndim != 3:
        raise ValueError("samples must have shape [conditions, samples, bins].")
    if (
        isinstance(noise_standard_deviation, bool)
        or not isinstance(noise_standard_deviation, (int, float))
        or not math.isfinite(float(noise_standard_deviation))
        or noise_standard_deviation <= 0.0
    ):
        raise ValueError("noise_standard_deviation must be finite and positive.")
    if (
        isinstance(w2_num_quantiles, bool)
        or not isinstance(w2_num_quantiles, int)
        or w2_num_quantiles < 1
    ):
        raise ValueError("w2_num_quantiles must be a positive integer.")
    if (
        isinstance(peak_threshold_fraction, bool)
        or not isinstance(peak_threshold_fraction, (int, float))
        or not math.isfinite(float(peak_threshold_fraction))
        or peak_threshold_fraction < 0.0
        or peak_threshold_fraction > 1.0
    ):
        raise ValueError("peak_threshold_fraction must lie in [0, 1].")

    samples = samples.to(torch.float64)
    targets = targets.to(torch.float64)
    measurements = measurements.to(torch.float64)
    kernel = kernel.to(torch.float64)
    log_rate_edges = log_rate_edges.to(torch.float64)

    repeated_targets = targets[:, None, :].expand_as(samples)
    sample_error = torch.linalg.vector_norm(samples - repeated_targets, dim=-1)
    target_norm = torch.linalg.vector_norm(repeated_targets, dim=-1).clamp_min(1.0e-12)
    sample_nrmse = sample_error / target_norm

    ensemble_mean = samples.mean(dim=1)
    ensemble_nrmse = torch.linalg.vector_norm(ensemble_mean - targets, dim=-1)
    ensemble_nrmse = ensemble_nrmse / torch.linalg.vector_norm(
        targets, dim=-1
    ).clamp_min(1.0e-12)

    sample_w1 = histogram_w1(samples, repeated_targets, log_rate_edges)
    ensemble_w1 = histogram_w1(ensemble_mean, targets, log_rate_edges)
    sample_w2 = histogram_w2(
        samples,
        repeated_targets,
        log_rate_edges,
        w2_num_quantiles,
    )
    ensemble_w2 = histogram_w2(
        ensemble_mean,
        targets,
        log_rate_edges,
        w2_num_quantiles,
    )

    predicted_measurements = torch.einsum("mn,bsn->bsm", kernel, samples)
    normalized_residual = (
        predicted_measurements - measurements[:, None, :]
    ) / float(noise_standard_deviation)
    forward_chi_square_per_measurement = normalized_residual.square().mean(dim=-1)

    sample_peak_count = _peak_count(samples, peak_threshold_fraction)
    target_peak_count = _peak_count(targets, peak_threshold_fraction)[:, None]
    peak_count_absolute_error = (sample_peak_count - target_peak_count).abs().to(
        torch.float64
    )

    log_rate_centers = 0.5 * (log_rate_edges[:-1] + log_rate_edges[1:])
    sample_dominant_centers = log_rate_centers[samples.argmax(dim=-1)]
    target_dominant_centers = log_rate_centers[targets.argmax(dim=-1)][:, None]

    mass_error = (samples.sum(dim=-1) - 1.0).abs()
    negative_fraction = (samples < 0).to(torch.float64).mean()

    metrics = {
        "sample_nrmse_mean": float(sample_nrmse.mean()),
        "ensemble_mean_nrmse": float(ensemble_nrmse.mean()),
        "sample_log_w1_mean": float(sample_w1.mean()),
        "ensemble_mean_log_w1": float(ensemble_w1.mean()),
        "sample_log_w2_mean": float(sample_w2.mean()),
        "ensemble_mean_log_w2": float(ensemble_w2.mean()),
        "forward_chi_square_per_measurement_mean": float(
            forward_chi_square_per_measurement.mean()
        ),
        "peak_count_absolute_error_mean": float(peak_count_absolute_error.mean()),
        "dominant_peak_log_rate_error_mean": float(
            (sample_dominant_centers - target_dominant_centers).abs().mean()
        ),
        "sample_mass_absolute_error_mean": float(mass_error.mean()),
        "sample_negative_fraction": float(negative_fraction),
    }
    return metrics


def _peak_count(
    histograms: torch.Tensor,
    threshold_fraction: float,
) -> torch.Tensor:
    if (
        isinstance(threshold_fraction, bool)
        or not isinstance(threshold_fraction, (int, float))
        or not math.isfinite(float(threshold_fraction))
        or threshold_fraction < 0.0
        or threshold_fraction > 1.0
    ):
        raise ValueError("threshold_fraction must lie in [0, 1].")
    maximum = histograms.max(dim=-1, keepdim=True).values
    threshold = float(threshold_fraction) * maximum
    negative_infinity = torch.full_like(histograms[..., :1], -torch.inf)
    left = torch.cat((negative_infinity, histograms[..., :-1]), dim=-1)
    right = torch.cat((histograms[..., 1:], negative_infinity), dim=-1)
    is_peak = (histograms > left) & (histograms >= right) & (histograms >= threshold)
    return is_peak.sum(dim=-1)


def find_latest_checkpoint(output_directory: Path, require_ema: bool) -> Path:
    """Resolve the committed latest state for this self-contained experiment."""

    from .checkpoint_utils import resolve_checkpoint

    return resolve_checkpoint(
        output_directory,
        require_ema=require_ema,
        allow_legacy_numbered=False,
    )


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_reconstruction_figure(
    path: Path,
    samples: torch.Tensor,
    target: torch.Tensor,
    measurement: torch.Tensor,
    rate_centers: torch.Tensor,
    echo_times: torch.Tensor,
    kernel: torch.Tensor,
    noise_standard_deviation: float,
    figure_title: str | None = None,
) -> None:
    if isinstance(noise_standard_deviation, bool) or not isinstance(
        noise_standard_deviation, (int, float)
    ):
        raise TypeError("noise_standard_deviation must be a real number.")
    if not math.isfinite(float(noise_standard_deviation)):
        raise ValueError("noise_standard_deviation must be finite.")
    if noise_standard_deviation <= 0.0:
        raise ValueError("noise_standard_deviation must be positive.")
    if figure_title is not None:
        if not isinstance(figure_title, str):
            raise TypeError("figure_title must be a string or None.")
        if not figure_title.strip():
            raise ValueError("figure_title must not be empty.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    samples = samples.detach().cpu().to(torch.float64)
    target = target.detach().cpu().to(torch.float64)
    measurement = measurement.detach().cpu().to(torch.float64)
    rate_centers = rate_centers.detach().cpu().to(torch.float64)
    echo_times = echo_times.detach().cpu().to(torch.float64)
    kernel = kernel.detach().cpu().to(torch.float64)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for index in range(samples.shape[0]):
        axes[0].plot(
            rate_centers.numpy(),
            samples[index].numpy(),
            color="#D6604D",
            alpha=0.30,
            linewidth=1.0,
        )
    axes[0].plot(
        rate_centers.numpy(),
        samples.mean(dim=0).numpy(),
        color="#D6604D",
        linewidth=2.0,
        label="reconstruction mean",
    )
    axes[0].plot(
        rate_centers.numpy(),
        target.numpy(),
        color="#2166AC",
        linewidth=2.0,
        label="ground-truth spectrum",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Relaxation rate $R_2$ (s$^{-1}$)")
    axes[0].set_ylabel("Bin mass")
    axes[0].set_title("Relaxation-rate spectrum")
    axes[0].legend(frameon=False)

    sample_decays = samples @ kernel.transpose(0, 1)
    target_decay = kernel @ target
    noise_band = float(noise_standard_deviation)
    sample_mean_decay = sample_decays.mean(dim=0)
    for index in range(sample_decays.shape[0]):
        axes[1].plot(
            (echo_times * 1.0e3).numpy(),
            sample_decays[index].numpy(),
            color="#D6604D",
            alpha=0.30,
            linewidth=1.0,
        )
    axes[1].fill_between(
        (echo_times * 1.0e3).numpy(),
        (target_decay - noise_band).numpy(),
        (target_decay + noise_band).numpy(),
        color="#F8E1D9",
        alpha=0.75,
        label=r"ground truth $\pm\sigma$",
    )
    axes[1].scatter(
        (echo_times * 1.0e3).numpy(),
        measurement.numpy(),
        s=16,
        facecolor="white",
        edgecolor="#D6604D",
        label="noisy decay measurement",
    )
    axes[1].plot(
        (echo_times * 1.0e3).numpy(),
        target_decay.numpy(),
        color="#2166AC",
        linewidth=2.0,
        label="ground-truth decay",
    )
    axes[1].plot(
        (echo_times * 1.0e3).numpy(),
        sample_mean_decay.numpy(),
        color="#D6604D",
        linewidth=2.0,
        label="reconstructed mean decay",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Echo time (ms)")
    axes[1].set_ylabel("Normalized echo amplitude")
    axes[1].set_title("Decay measurement and forward reconstruction")
    axes[1].legend(frameon=False)

    if figure_title is not None:
        figure.suptitle(figure_title)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".png",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, dpi=180)
        os.replace(temporary_path, path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def save_reconstruction_grid_figure(
    path: Path,
    samples: torch.Tensor,
    targets: torch.Tensor,
    rate_centers: torch.Tensor,
) -> None:
    """Atomically save 16 spectrum reconstructions as an axis-free 4x4 grid.

    The first tensor dimension indexes distinct measurement conditions and the
    second indexes independent reconstruction draws for the same condition.
    Exactly the first 16 conditions are visualized; additional conditions are
    intentionally ignored so that every saved grid has an identical layout.
    """

    if samples.ndim != 3:
        raise ValueError(
            "samples must have shape [conditions, samples, spectrum_bins]."
        )
    if targets.ndim != 2:
        raise ValueError("targets must have shape [conditions, spectrum_bins].")
    if rate_centers.ndim != 1:
        raise ValueError("rate_centers must be one-dimensional.")
    if samples.shape[0] != targets.shape[0]:
        raise ValueError("samples and targets must have the same condition count.")
    if samples.shape[0] < 16:
        raise ValueError(
            "samples and targets must contain at least 16 conditions for the "
            "4x4 reconstruction grid."
        )
    if samples.shape[1] < 1:
        raise ValueError("Each condition must contain at least one sample.")
    if (
        samples.shape[2] != targets.shape[1]
        or samples.shape[2] != rate_centers.shape[0]
    ):
        raise ValueError(
            "samples, targets, and rate_centers must use the same spectrum-bin "
            "count."
        )
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("samples must contain only finite values.")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets must contain only finite values.")
    if (
        not bool(torch.isfinite(rate_centers).all())
        or bool((rate_centers <= 0).any())
        or bool((torch.diff(rate_centers) <= 0).any())
    ):
        raise ValueError(
            "rate_centers must be finite, positive, and strictly increasing."
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    samples = samples[:16].detach().cpu().to(torch.float64)
    targets = targets[:16].detach().cpu().to(torch.float64)
    rate_centers = rate_centers.detach().cpu().to(torch.float64)
    reconstruction_means = samples.mean(dim=1)

    figure, axes = plt.subplots(4, 4, figsize=(8.0, 8.0))
    temporary_path = None
    try:
        for condition_index, axis in enumerate(axes.flat):
            axis.plot(
                rate_centers.numpy(),
                reconstruction_means[condition_index].numpy(),
                color="#D6604D",
                linewidth=2.0,
            )
            axis.plot(
                rate_centers.numpy(),
                targets[condition_index].numpy(),
                color="#2166AC",
                linewidth=2.0,
            )
            axis.set_xscale("log")
            axis.set_xticks(())
            axis.set_yticks(())
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_title("")
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_axis_off()
        figure.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.99,
            wspace=0.03,
            hspace=0.03,
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".png",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        figure.savefig(temporary_path, dpi=180)
        os.replace(temporary_path, path)
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
