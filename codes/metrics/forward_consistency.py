"""Forward errors of individual draws and of their arithmetic mean spectrum.

These are signal diagnostics, not proper posterior scores. The two aggregations
are kept separate: averaging squared draw errors differs from scoring the mean.
No noise is added and no spectra are selected or renormalized.
"""

from __future__ import annotations

import math

import torch


FORWARD_CONSISTENCY_PROTOCOL = "saved_joint_pairs_forward_errors_v1"
METRICS = ("noisy_mse", "clean_mse", "noisy_normalized_mse", "clean_normalized_mse")
MEAN_SPECTRUM_PROTOCOL = "arithmetic_mean_spectrum_forward_errors_v1"
MEAN_SPECTRUM_METRICS = tuple("mean_spectrum_" + name for name in METRICS)


def _finite_float(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values.")


def per_sample_forward_errors(
    samples: torch.Tensor,
    truth_histograms: torch.Tensor,
    measurements: torch.Tensor,
    kernel: torch.Tensor,
    *,
    noise_standard_deviation: float,
    condition_batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Return [G,K] MSEs, with the mean taken over M echo measurements only.

    Noisy: ||A h_hat - y||^2/M. Clean: ||A(h_hat-h)||^2/M, exactly the
    same-noise comparison after cancellation. The clean calculation uses the
    saved physical kernel and truth spectrum in float64, avoiding float32
    rounding of the stored clean signal. A true-spectrum prediction has zero
    clean error. Unit histogram mass is checked, never repaired silently.
    """
    for value, name in ((samples, "samples"), (truth_histograms, "truth_histograms"),
                        (measurements, "measurements"), (kernel, "kernel")):
        _finite_float(value, name)
        if value.device.type != "cpu":
            raise ValueError("Offline forward evaluation expects CPU tensors.")
    if samples.ndim != 3 or min(samples.shape) < 1:
        raise ValueError("samples must have shape [G,K,N] with nonempty axes.")
    groups, draws, bins = samples.shape
    if truth_histograms.shape != (groups, bins):
        raise ValueError("truth_histograms must have shape [G,N].")
    if measurements.ndim != 2 or measurements.shape[0] != groups or measurements.shape[1] < 1:
        raise ValueError("measurements must have shape [G,M].")
    if kernel.shape != (measurements.shape[1], bins):
        raise ValueError("kernel must have shape [M,N].")
    sigma = noise_standard_deviation
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("noise_standard_deviation must be finite and positive.")
    if isinstance(condition_batch_size, bool) or not isinstance(condition_batch_size, int) or condition_batch_size < 1:
        raise ValueError("condition_batch_size must be a positive integer.")
    for value, name in ((samples, "samples"), (truth_histograms, "truth_histograms")):
        mass = value.double().sum(-1)
        if (value < 0).any() or not torch.allclose(mass, torch.ones_like(mass), atol=2e-5, rtol=0):
            raise ValueError(f"{name} must contain nonnegative unit-mass histograms.")

    noisy = torch.empty((groups, draws), dtype=torch.float64)
    clean = torch.empty_like(noisy)
    operator = kernel.double().T
    for start in range(0, groups, condition_batch_size):
        stop = min(start + condition_batch_size, groups)
        forecast = samples[start:stop].double()
        truth = truth_histograms[start:stop, None, :].double()
        observed = measurements[start:stop, None, :].double()
        noisy[start:stop] = (forecast @ operator - observed).square().mean(-1)
        clean[start:stop] = ((forecast - truth) @ operator).square().mean(-1)
    return {
        "noisy_mse": noisy,
        "clean_mse": clean,
        "noisy_normalized_mse": noisy / sigma**2,
        "clean_normalized_mse": clean / sigma**2,
    }


def mean_spectrum_forward_errors(
    samples: torch.Tensor,
    truth_histograms: torch.Tensor,
    measurements: torch.Tensor,
    kernel: torch.Tensor,
    *,
    noise_standard_deviation: float,
    condition_batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Score the arithmetic mean of all saved draws; return [G,1] errors.

    Validate every input draw before averaging so a malformed ensemble cannot
    hide negative mass or invalid normalization. Mean in float64, including
    for deterministic models with a single saved draw.
    """
    _finite_float(samples, "samples")
    if samples.ndim != 3 or min(samples.shape) < 1:
        raise ValueError("samples must have shape [G,K,N] with nonempty axes.")
    mass = samples.double().sum(-1)
    if (samples < 0).any() or not torch.allclose(mass, torch.ones_like(mass), atol=2e-5, rtol=0):
        raise ValueError("samples must contain nonnegative unit-mass histograms.")
    errors = per_sample_forward_errors(
        samples.double().mean(1, keepdim=True), truth_histograms, measurements, kernel,
        noise_standard_deviation=noise_standard_deviation,
        condition_batch_size=condition_batch_size,
    )
    return {"mean_spectrum_" + name: value for name, value in errors.items()}


def summarize_forward_errors(errors: dict[str, torch.Tensor]) -> dict:
    """Average draws within a condition, then give all conditions equal weight.

    Standard errors and approximate normal 95% confidence intervals use G
    independent conditions, not G*K purported independent test observations.
    Sample quantiles describe the pooled draw errors, not confidence bounds.
    """
    result = {}
    for name in (*METRICS, *MEAN_SPECTRUM_METRICS):
        if name not in errors:
            continue
        values = errors[name]
        per_condition = values.mean(dim=1)
        groups = len(per_condition)
        mean = float(per_condition.mean())
        sd = float(per_condition.std(correction=1)) if groups > 1 else None
        se = sd / math.sqrt(groups) if sd is not None else None
        quantiles = torch.quantile(values.flatten(), torch.tensor([.05, .5, .95], dtype=torch.float64))
        result[name] = {
            "mean": mean,
            "condition_standard_deviation": sd,
            "condition_standard_error": se,
            "condition_mean_ci95_normal": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
            "pooled_sample_q05": float(quantiles[0]),
            "pooled_sample_median": float(quantiles[1]),
            "pooled_sample_q95": float(quantiles[2]),
            "root_mean_mse": math.sqrt(mean),
        }
    return result
