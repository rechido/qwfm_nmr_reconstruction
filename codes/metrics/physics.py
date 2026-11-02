"""Forward-model and derived-quantity diagnostics for NMR posterior samples."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

import torch

from .distances import validate_log_rate_edges


DEFAULT_RATE_WINDOWS: Final[dict[str, tuple[float, float]]] = {
    "rate_2_10": (2.0, 10.0),
    "rate_10_50": (10.0, 50.0),
    "rate_50_250": (50.0, 250.0),
}


def validate_nonnegative_spectra(
    samples: torch.Tensor, *, require_unit_mass: bool = False
) -> None:
    """Validate a bank of nonnegative spectra, optionally on the simplex."""

    if not isinstance(samples, torch.Tensor) or samples.ndim != 2:
        raise ValueError("samples must have shape [S,N].")
    if samples.shape[0] < 1 or samples.shape[1] < 2:
        raise ValueError("samples must be nonempty and have at least two bins.")
    if not samples.is_floating_point():
        raise TypeError("samples must have a floating-point dtype.")
    if not bool(torch.isfinite(samples).all()) or bool((samples < 0).any()):
        raise ValueError("samples must be finite and nonnegative.")
    if require_unit_mass:
        mass = samples.sum(dim=-1)
        if not bool(
            torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5)
        ):
            raise ValueError("every sample must have unit mass.")


def _validate_samples(samples: torch.Tensor) -> None:
    validate_nonnegative_spectra(samples, require_unit_mass=True)


def forward_reduced_chi_square(
    samples: torch.Tensor,
    measurement: torch.Tensor,
    kernel: torch.Tensor,
    *,
    noise_standard_deviation: float,
    require_unit_mass: bool = True,
) -> torch.Tensor:
    """Return likelihood misfit per sample divided by measurement count.

    This is a posterior forward-consistency diagnostic, not a statistic that is
    forced to equal one after conditioning on the observed measurement.
    """

    validate_nonnegative_spectra(samples, require_unit_mass=require_unit_mass)
    if not isinstance(measurement, torch.Tensor) or measurement.ndim != 1:
        raise ValueError("measurement must have shape [M].")
    if not isinstance(kernel, torch.Tensor) or kernel.ndim != 2:
        raise ValueError("kernel must have shape [M,N].")
    if tuple(kernel.shape) != (int(measurement.shape[0]), int(samples.shape[1])):
        raise ValueError("kernel dimensions must match measurement and samples.")
    if samples.device != measurement.device or samples.device != kernel.device:
        raise ValueError("samples, measurement, and kernel must share a device.")
    if not bool(torch.isfinite(measurement).all()) or not bool(
        torch.isfinite(kernel).all()
    ):
        raise ValueError("measurement and kernel must be finite.")
    if (
        isinstance(noise_standard_deviation, bool)
        or not isinstance(noise_standard_deviation, (float, int))
        or not math.isfinite(float(noise_standard_deviation))
        or float(noise_standard_deviation) <= 0
    ):
        raise ValueError("noise_standard_deviation must be finite and positive.")
    work_samples = samples.to(torch.float64)
    prediction = work_samples @ kernel.to(torch.float64).T
    standardized_residual = (
        prediction - measurement.to(torch.float64)[None, :]
    ) / float(noise_standard_deviation)
    return standardized_residual.square().mean(dim=-1)


def forward_residual_diagnostics(
    samples: torch.Tensor,
    measurement: torch.Tensor,
    kernel: torch.Tensor,
    *,
    noise_standard_deviation: float,
    require_unit_mass: bool = True,
) -> dict[str, float]:
    """Summarize forward likelihood consistency over posterior samples."""

    reduced = forward_reduced_chi_square(
        samples,
        measurement,
        kernel,
        noise_standard_deviation=noise_standard_deviation,
        require_unit_mass=require_unit_mass,
    )
    prediction = samples.to(torch.float64) @ kernel.to(torch.float64).T
    posterior_mean_residual = (
        prediction.mean(dim=0) - measurement.to(torch.float64)
    ) / float(noise_standard_deviation)
    quantiles = torch.quantile(
        reduced,
        torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64, device=reduced.device),
    )
    return {
        "forward_reduced_chi_square_mean": float(reduced.mean().item()),
        "forward_reduced_chi_square_q05": float(quantiles[0].item()),
        "forward_reduced_chi_square_median": float(quantiles[1].item()),
        "forward_reduced_chi_square_q95": float(quantiles[2].item()),
        "posterior_mean_residual_rms_in_sigma": float(
            posterior_mean_residual.square().mean().sqrt().item()
        ),
        "posterior_mean_residual_max_abs_in_sigma": float(
            posterior_mean_residual.abs().max().item()
        ),
    }


def _validated_rate_windows(
    rate_windows: Mapping[str, tuple[float, float]] | None,
    log_rate_edges: torch.Tensor,
) -> dict[str, tuple[float, float]]:
    resolved = dict(DEFAULT_RATE_WINDOWS if rate_windows is None else rate_windows)
    domain_low = float(log_rate_edges[0].exp().item())
    domain_high = float(log_rate_edges[-1].exp().item())
    for name, bounds in resolved.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each rate window must have a nonempty string name.")
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(f"rate window {name!r} must contain two bounds.")
        low, high = float(bounds[0]), float(bounds[1])
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or low >= high:
            raise ValueError(f"rate window {name!r} has invalid positive bounds.")
        tolerance = 1.0e-10 * max(1.0, domain_high)
        if low < domain_low - tolerance or high > domain_high + tolerance:
            raise ValueError(
                f"rate window {name!r} lies outside the reconstruction grid."
            )
        resolved[name] = (max(low, domain_low), min(high, domain_high))
    return resolved


def rate_window_fractions(
    samples: torch.Tensor,
    log_rate_edges: torch.Tensor,
    *,
    rate_windows: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, torch.Tensor]:
    """Integrate histogram mass over predeclared physical rate windows.

    Boundary-bin mass is split by overlap length in natural-log-rate, matching
    the piecewise-uniform density convention used by W1 and W2.
    """

    _validate_samples(samples)
    edges = validate_log_rate_edges(log_rate_edges, num_bins=int(samples.shape[1]))
    if edges.device != samples.device:
        raise ValueError("samples and log_rate_edges must share a device.")
    work_edges = edges.to(torch.float64)
    widths = torch.diff(work_edges)
    windows = _validated_rate_windows(rate_windows, work_edges)
    normalized_samples = samples.to(torch.float64)
    normalized_samples /= normalized_samples.sum(dim=-1, keepdim=True)
    fractions: dict[str, torch.Tensor] = {}
    for name, (low, high) in windows.items():
        low_log = math.log(low)
        high_log = math.log(high)
        low_tensor = torch.tensor(
            low_log, device=work_edges.device, dtype=work_edges.dtype
        )
        high_tensor = torch.tensor(
            high_log, device=work_edges.device, dtype=work_edges.dtype
        )
        overlap = (
            torch.minimum(work_edges[1:], high_tensor)
            - torch.maximum(work_edges[:-1], low_tensor)
        ).clamp_min(0.0)
        fractions[name] = normalized_samples @ (overlap / widths)
    return fractions


def derived_quantities(
    samples: torch.Tensor,
    log_rate_edges: torch.Tensor,
    *,
    rate_windows: Mapping[str, tuple[float, float]] | None = None,
    peak_threshold_fraction: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Calculate fixed-window fractions, rate means, and effective peak count."""

    _validate_samples(samples)
    edges = validate_log_rate_edges(log_rate_edges, num_bins=int(samples.shape[1]))
    if edges.device != samples.device:
        raise ValueError("samples and log_rate_edges must share a device.")
    if (
        isinstance(peak_threshold_fraction, bool)
        or not isinstance(peak_threshold_fraction, (float, int))
        or not math.isfinite(float(peak_threshold_fraction))
        or not 0 <= float(peak_threshold_fraction) <= 1
    ):
        raise ValueError("peak_threshold_fraction must lie in [0,1].")

    quantities = rate_window_fractions(samples, edges, rate_windows=rate_windows)
    work = samples.to(torch.float64)
    work /= work.sum(dim=-1, keepdim=True)
    work_edges = edges.to(torch.float64)
    log_centres = 0.5 * (work_edges[:-1] + work_edges[1:])
    # E[R | log R uniform in a bin] is the logarithmic mean of its endpoints.
    rate_edges = work_edges.exp()
    mean_rate_in_bin = torch.diff(rate_edges) / torch.diff(work_edges)
    quantities["mean_log_rate"] = work @ log_centres
    quantities["mean_rate"] = work @ mean_rate_in_bin

    negative_infinity = torch.full(
        (work.shape[0], 1), -torch.inf, device=work.device, dtype=work.dtype
    )
    left = torch.cat((negative_infinity, work[:, :-1]), dim=1)
    right = torch.cat((work[:, 1:], negative_infinity), dim=1)
    threshold = float(peak_threshold_fraction) * work.max(dim=1, keepdim=True).values
    # Asymmetric tie handling counts the leftmost point of a flat plateau once.
    peaks = (work > left) & (work >= right) & (work >= threshold) & (work > 0)
    quantities["effective_peak_count"] = peaks.sum(dim=1).to(torch.float64)
    return quantities


def summarize_derived_quantities(
    quantities: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Convert sample-level derived quantities into JSON-safe summaries."""

    result: dict[str, dict[str, float]] = {}
    for name, values in quantities.items():
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 1
            or values.numel() < 1
        ):
            raise ValueError(f"derived quantity {name!r} must be a nonempty vector.")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"derived quantity {name!r} contains non-finite values.")
        work = values.to(torch.float64)
        quantiles = torch.quantile(
            work,
            torch.tensor([0.05, 0.5, 0.95], device=work.device, dtype=torch.float64),
        )
        result[name] = {
            "mean": float(work.mean().item()),
            "standard_deviation": float(work.std(unbiased=False).item()),
            "q05": float(quantiles[0].item()),
            "median": float(quantiles[1].item()),
            "q95": float(quantiles[2].item()),
        }
    return result


def credible_interval_calibration(
    sampled_values: torch.Tensor,
    truth_values: torch.Tensor,
    *,
    level: float = 0.90,
) -> dict[str, torch.Tensor | float]:
    """Evaluate equal-tail interval coverage across simulated conditions.

    ``sampled_values`` has shape ``[G,K,D]`` (or ``[G,K]``) and truth has
    shape ``[G,D]`` (or ``[G]``).  The returned coverage is calibration across
    conditions and is distinct from the PRDC metric also named ``coverage``.
    """

    if not isinstance(sampled_values, torch.Tensor) or sampled_values.ndim not in (
        2,
        3,
    ):
        raise ValueError("sampled_values must have shape [G,K] or [G,K,D].")
    if not isinstance(truth_values, torch.Tensor):
        raise TypeError("truth_values must be a torch.Tensor.")
    if sampled_values.ndim == 2:
        sampled_values = sampled_values[..., None]
        truth_values = truth_values[..., None]
    if truth_values.ndim != 2 or tuple(truth_values.shape) != (
        int(sampled_values.shape[0]),
        int(sampled_values.shape[2]),
    ):
        raise ValueError("truth_values dimensions must match sampled_values [G,*,D].")
    if not bool(torch.isfinite(sampled_values).all()) or not bool(
        torch.isfinite(truth_values).all()
    ):
        raise ValueError("sampled and truth values must be finite.")
    if (
        not isinstance(level, (float, int))
        or isinstance(level, bool)
        or not 0 < level < 1
    ):
        raise ValueError("level must lie strictly between zero and one.")
    work = sampled_values.to(torch.float64)
    alpha = 0.5 * (1.0 - float(level))
    bounds = torch.quantile(
        work,
        torch.tensor([alpha, 1.0 - alpha], device=work.device, dtype=torch.float64),
        dim=1,
    )
    truth = truth_values.to(device=work.device, dtype=torch.float64)
    covered = (truth >= bounds[0]) & (truth <= bounds[1])
    width = bounds[1] - bounds[0]
    return {
        "nominal_level": float(level),
        "coverage_by_quantity": covered.to(torch.float64).mean(dim=0),
        "mean_width_by_quantity": width.mean(dim=0),
        "lower_bounds": bounds[0],
        "upper_bounds": bounds[1],
        "covered": covered,
    }


__all__ = [
    "DEFAULT_RATE_WINDOWS",
    "credible_interval_calibration",
    "derived_quantities",
    "forward_reduced_chi_square",
    "forward_residual_diagnostics",
    "rate_window_fractions",
    "summarize_derived_quantities",
    "validate_nonnegative_spectra",
]
