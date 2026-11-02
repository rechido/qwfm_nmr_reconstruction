from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from ..utils.duplicate_utils import exact_duplicate_diagnostics
from .grid import NMRGrid, build_nmr_grid
from .simulator import (
    GammaMixtureBatch,
    gamma_component_bin_masses,
    keyed_seed,
    private_generator,
    sample_gamma_mixture_prior,
)
from .spec import ContinuousNMRSpec


@dataclass(frozen=True)
class SMCSettings:
    sampler_version: str = "tempering_adaptation_v2"
    num_particles: int = 4096
    target_conditional_ess_fraction: float = 0.95
    resample_ess_fraction: float = 0.5
    rejuvenation_sweeps: int = 25
    tempering_adaptation_fraction: float = 0.5
    final_rejuvenation_sweeps: int = 10
    maximum_final_rejuvenation_sweeps: int = 128
    final_diagnostic_interval: int = 5
    final_acceptance_window: int = 5
    final_minimum_mean_acceptance: float = 0.10
    final_minimum_unique_fraction: float = 0.999
    final_maximum_k3_zero_radius_fraction: float = 0.0
    maximum_temperatures: int = 512
    beta_bisection_iterations: int = 48
    center_logit_step: float = 0.15
    log_cv_step_fraction: float = 0.05
    weight_logit_step: float = 0.25
    adaptation_target_acceptance: float = 0.30
    adaptation_gain: float = 0.50
    minimum_step_scale: float = 1.0e-6
    maximum_step_scale: float = 2.0
    output_samples: int = 4096

    def __post_init__(self) -> None:
        if self.sampler_version != "tempering_adaptation_v2":
            raise ValueError("Unsupported SMC sampler_version.")
        integer_names = (
            "num_particles",
            "rejuvenation_sweeps",
            "final_rejuvenation_sweeps",
            "maximum_final_rejuvenation_sweeps",
            "final_diagnostic_interval",
            "final_acceptance_window",
            "maximum_temperatures",
            "beta_bisection_iterations",
            "output_samples",
        )
        for name in integer_names:
            value = getattr(self, name)
            minimum = 0 if name == "rejuvenation_sweeps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if self.num_particles <= 3:
            raise ValueError(
                "num_particles must exceed three for the k=3 duplicate gate."
            )
        if self.output_samples > self.num_particles:
            raise ValueError(
                "output_samples must not exceed num_particles; replacement sampling "
                "would create exact duplicates in the compiled reference."
            )
        if self.output_samples <= 3:
            raise ValueError(
                "output_samples must exceed three for the k=3 duplicate diagnostic."
            )
        if self.maximum_final_rejuvenation_sweeps < self.final_rejuvenation_sweeps:
            raise ValueError(
                "maximum_final_rejuvenation_sweeps must not be below "
                "final_rejuvenation_sweeps."
            )
        if self.final_acceptance_window > self.maximum_final_rejuvenation_sweeps:
            raise ValueError(
                "final_acceptance_window must not exceed "
                "maximum_final_rejuvenation_sweeps."
            )
        if not 0.0 < self.target_conditional_ess_fraction < 1.0:
            raise ValueError("target_conditional_ess_fraction must lie in (0,1).")
        if not 0.0 <= self.tempering_adaptation_fraction < 1.0:
            raise ValueError("tempering_adaptation_fraction must lie in [0,1).")
        if not 0.0 < self.resample_ess_fraction < self.target_conditional_ess_fraction:
            raise ValueError(
                "resample_ess_fraction must be positive and below the CESS target."
            )
        for name in (
            "center_logit_step",
            "log_cv_step_fraction",
            "weight_logit_step",
            "adaptation_gain",
            "minimum_step_scale",
            "maximum_step_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 < self.adaptation_target_acceptance < 1.0:
            raise ValueError("adaptation_target_acceptance must lie in (0,1).")
        if not 0.0 <= self.final_minimum_mean_acceptance < 1.0:
            raise ValueError("final_minimum_mean_acceptance must lie in [0,1).")
        if not 0.0 < self.final_minimum_unique_fraction <= 1.0:
            raise ValueError("final_minimum_unique_fraction must lie in (0,1].")
        if not 0.0 <= self.final_maximum_k3_zero_radius_fraction <= 1.0:
            raise ValueError("final_maximum_k3_zero_radius_fraction must lie in [0,1].")
        if self.maximum_step_scale < self.minimum_step_scale:
            raise ValueError("maximum_step_scale must not be below minimum_step_scale.")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _ParticleState:
    center_logits: torch.Tensor
    log_coefficients_of_variation: torch.Tensor
    weight_logits: torch.Tensor
    log_likelihood: torch.Tensor
    log_transformed_prior: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "_ParticleState":
        return _ParticleState(
            center_logits=self.center_logits.index_select(0, indices),
            log_coefficients_of_variation=self.log_coefficients_of_variation.index_select(
                0, indices
            ),
            weight_logits=self.weight_logits.index_select(0, indices),
            log_likelihood=self.log_likelihood.index_select(0, indices),
            log_transformed_prior=self.log_transformed_prior.index_select(0, indices),
        )


@dataclass
class FixedJSMCResult:
    component_count: int
    histograms: torch.Tensor
    centers: torch.Tensor
    coefficients_of_variation: torch.Tensor
    mixture_weights: torch.Tensor
    retained_masses: torch.Tensor
    log_evidence: float
    diagnostics: dict[str, Any]


@dataclass
class ReferenceConditionResult:
    reference_histograms: torch.Tensor
    reference_component_counts: torch.Tensor
    reference_active_mask: torch.Tensor
    reference_centers: torch.Tensor
    reference_coefficients_of_variation: torch.Tensor
    reference_mixture_weights: torch.Tensor
    posterior_component_count_probability: torch.Tensor
    log_evidence: torch.Tensor
    diagnostics: dict[str, Any]


def _simplex_to_alr(simplex: torch.Tensor) -> torch.Tensor:
    if simplex.shape[-1] == 1:
        return simplex.new_empty(simplex.shape[:-1] + (0,))
    return simplex[..., :-1].log() - simplex[..., -1:].log()


def _alr_to_simplex(logits: torch.Tensor) -> torch.Tensor:
    reference = torch.zeros(
        logits.shape[:-1] + (1,), dtype=logits.dtype, device=logits.device
    )
    return torch.softmax(torch.cat((logits, reference), dim=-1), dim=-1)


def _ordered_unit_to_logits(ordered_unit: torch.Tensor) -> torch.Tensor:
    leading = ordered_unit[..., :1]
    middle = torch.diff(ordered_unit, dim=-1)
    trailing = 1.0 - ordered_unit[..., -1:]
    spacings = torch.cat((leading, middle, trailing), dim=-1)
    if bool((spacings <= 0.0).any()):
        raise ValueError("Ordered centers must be strictly inside the center interval.")
    return spacings[..., :-1].log() - spacings[..., -1:].log()


def _center_logits_to_ordered_unit(
    center_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    spacings = _alr_to_simplex(center_logits)
    ordered = torch.cumsum(spacings, dim=-1)[..., :-1]
    return ordered, spacings


def _decode_state_parameters(
    state: _ParticleState,
    component_count: int,
    spec: ContinuousNMRSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ordered_unit, center_spacings = _center_logits_to_ordered_unit(state.center_logits)
    log_center_minimum = math.log(spec.center_minimum)
    log_center_range = math.log(spec.center_maximum / spec.center_minimum)
    centers = torch.exp(log_center_minimum + log_center_range * ordered_unit)
    cvs = state.log_coefficients_of_variation.exp()
    weights = _alr_to_simplex(state.weight_logits)
    if centers.shape[-1] != component_count or cvs.shape[-1] != component_count:
        raise RuntimeError("Decoded SMC dimension does not match component_count.")
    return centers, cvs, weights, center_spacings


def _transformed_prior_log_density(
    center_spacings: torch.Tensor,
    weights: torch.Tensor,
    *,
    dirichlet_concentration: float,
) -> torch.Tensor:
    # Ordered log-centers have constant density.  The ALR transformation from
    # center spacings contributes product(spacing).  A Dirichlet(alpha) density
    # plus the simplex ALR Jacobian contributes alpha * sum(log weight).
    return center_spacings.log().sum(dim=-1) + float(
        dirichlet_concentration
    ) * weights.log().sum(dim=-1)


def _full_gaussian_log_likelihood(
    measurements: torch.Tensor,
    clean_measurements: torch.Tensor,
    noise_standard_deviation: float,
) -> torch.Tensor:
    residual = (measurements[None, :] - clean_measurements) / noise_standard_deviation
    dimension = int(measurements.numel())
    normalization = dimension * (
        math.log(noise_standard_deviation) + 0.5 * math.log(2.0 * math.pi)
    )
    return -0.5 * residual.square().sum(dim=-1) - normalization


def _evaluate_particle_state(
    state: _ParticleState,
    *,
    component_count: int,
    measurement: torch.Tensor,
    spec: ContinuousNMRSpec,
    grid: NMRGrid,
) -> tuple[
    _ParticleState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    centers, cvs, weights, center_spacings = _decode_state_parameters(
        state, component_count, spec
    )
    component_masses, retained = gamma_component_bin_masses(
        centers, cvs, grid.rate_edges
    )
    valid = (retained >= spec.minimum_component_retained_mass).all(dim=-1)
    safe_retained = retained.clamp_min(torch.finfo(retained.dtype).tiny)
    components = component_masses / safe_retained[:, :, None]
    histograms = (weights[:, :, None] * components).sum(dim=1)
    histograms = histograms.clamp_min(0.0)
    histograms = histograms / histograms.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(histograms.dtype).tiny
    )
    clean = histograms @ grid.kernel.T
    transformed_prior = _transformed_prior_log_density(
        center_spacings,
        weights,
        dirichlet_concentration=spec.mixture_weight_concentration,
    )
    likelihood = _full_gaussian_log_likelihood(
        measurement, clean, spec.noise_standard_deviation
    )
    negative_infinity = torch.full_like(likelihood, -torch.inf)
    likelihood = torch.where(valid, likelihood, negative_infinity)
    transformed_prior = torch.where(valid, transformed_prior, negative_infinity)
    evaluated = _ParticleState(
        center_logits=state.center_logits,
        log_coefficients_of_variation=state.log_coefficients_of_variation,
        weight_logits=state.weight_logits,
        log_likelihood=likelihood,
        log_transformed_prior=transformed_prior,
    )
    return evaluated, histograms, centers, cvs, weights, retained


def _initial_state(
    mixture: GammaMixtureBatch,
    *,
    component_count: int,
    spec: ContinuousNMRSpec,
    measurement: torch.Tensor,
    grid: NMRGrid,
    device: torch.device,
) -> _ParticleState:
    centers = mixture.centers[:, :component_count].to(
        device=device, dtype=torch.float64
    )
    cvs = mixture.coefficients_of_variation[:, :component_count].to(
        device=device, dtype=torch.float64
    )
    weights = mixture.mixture_weights[:, :component_count].to(
        device=device, dtype=torch.float64
    )
    log_minimum = math.log(spec.center_minimum)
    log_range = math.log(spec.center_maximum / spec.center_minimum)
    ordered_unit = (centers.log() - log_minimum) / log_range
    provisional = _ParticleState(
        center_logits=_ordered_unit_to_logits(ordered_unit),
        log_coefficients_of_variation=cvs.log(),
        weight_logits=_simplex_to_alr(weights),
        log_likelihood=torch.empty(
            centers.shape[0], dtype=torch.float64, device=device
        ),
        log_transformed_prior=torch.empty(
            centers.shape[0], dtype=torch.float64, device=device
        ),
    )
    evaluated, *_ = _evaluate_particle_state(
        provisional,
        component_count=component_count,
        measurement=measurement,
        spec=spec,
        grid=grid,
    )
    if not bool(torch.isfinite(evaluated.log_likelihood).all()):
        raise RuntimeError("A retained-mass-conditioned initial particle is invalid.")
    return evaluated


def _log_conditional_ess(
    normalized_log_weights: torch.Tensor,
    log_likelihood: torch.Tensor,
    delta_beta: float,
) -> torch.Tensor:
    increment = float(delta_beta) * log_likelihood
    log_first = torch.logsumexp(normalized_log_weights + increment, dim=0)
    log_second = torch.logsumexp(normalized_log_weights + 2.0 * increment, dim=0)
    return math.log(normalized_log_weights.numel()) + 2.0 * log_first - log_second


def choose_next_beta(
    beta: float,
    normalized_log_weights: torch.Tensor,
    log_likelihood: torch.Tensor,
    *,
    target_fraction: float,
    bisection_iterations: int,
) -> float:
    if beta >= 1.0:
        return 1.0
    target_log_ess = math.log(target_fraction * normalized_log_weights.numel())
    remaining = 1.0 - beta
    if (
        float(_log_conditional_ess(normalized_log_weights, log_likelihood, remaining))
        >= target_log_ess
    ):
        return 1.0
    lower = 0.0
    upper = remaining
    for _ in range(bisection_iterations):
        midpoint = 0.5 * (lower + upper)
        value = float(
            _log_conditional_ess(normalized_log_weights, log_likelihood, midpoint)
        )
        if value < target_log_ess:
            upper = midpoint
        else:
            lower = midpoint
    delta = max(lower, torch.finfo(torch.float64).eps)
    return min(1.0, beta + delta)


def _effective_sample_size(normalized_log_weights: torch.Tensor) -> float:
    return float(torch.exp(-torch.logsumexp(2.0 * normalized_log_weights, dim=0)))


def systematic_resample(
    normalized_log_weights: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    count = int(normalized_log_weights.numel())
    weights = normalized_log_weights.exp()
    offset = (
        torch.rand((), generator=generator, dtype=weights.dtype, device=weights.device)
        / count
    )
    positions = (
        offset + torch.arange(count, dtype=weights.dtype, device=weights.device) / count
    )
    cumulative = weights.cumsum(dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=False).to(torch.int64)


def _reflect(values: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    width = upper - lower
    if width <= 0.0:
        raise ValueError("Reflection bounds must be ordered.")
    phase = torch.remainder(values - lower, 2.0 * width)
    return lower + torch.where(phase <= width, phase, 2.0 * width - phase)


def _metropolis_sweep(
    state: _ParticleState,
    *,
    beta: float,
    component_count: int,
    measurement: torch.Tensor,
    spec: ContinuousNMRSpec,
    grid: NMRGrid,
    center_step: float,
    log_cv_step: float,
    weight_step: float,
    generator: torch.Generator,
) -> tuple[_ParticleState, float]:
    proposed_center = state.center_logits + center_step * torch.randn(
        state.center_logits.shape,
        dtype=state.center_logits.dtype,
        device=state.center_logits.device,
        generator=generator,
    )
    proposed_log_cv = state.log_coefficients_of_variation + log_cv_step * torch.randn(
        state.log_coefficients_of_variation.shape,
        dtype=state.log_coefficients_of_variation.dtype,
        device=state.log_coefficients_of_variation.device,
        generator=generator,
    )
    proposed_log_cv = _reflect(
        proposed_log_cv,
        math.log(spec.coefficient_of_variation_minimum),
        math.log(spec.coefficient_of_variation_maximum),
    )
    proposed_weight = state.weight_logits + weight_step * torch.randn(
        state.weight_logits.shape,
        dtype=state.weight_logits.dtype,
        device=state.weight_logits.device,
        generator=generator,
    )
    proposal = _ParticleState(
        center_logits=proposed_center,
        log_coefficients_of_variation=proposed_log_cv,
        weight_logits=proposed_weight,
        log_likelihood=state.log_likelihood,
        log_transformed_prior=state.log_transformed_prior,
    )
    proposal, *_ = _evaluate_particle_state(
        proposal,
        component_count=component_count,
        measurement=measurement,
        spec=spec,
        grid=grid,
    )
    current_target = state.log_transformed_prior + float(beta) * state.log_likelihood
    proposed_target = (
        proposal.log_transformed_prior + float(beta) * proposal.log_likelihood
    )
    log_acceptance = proposed_target - current_target
    uniforms = torch.rand(
        log_acceptance.shape,
        dtype=log_acceptance.dtype,
        device=log_acceptance.device,
        generator=generator,
    ).clamp_min(torch.finfo(log_acceptance.dtype).tiny)
    accepted = uniforms.log() < log_acceptance

    def select(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        shape = (accepted.shape[0],) + (1,) * (new.ndim - 1)
        return torch.where(accepted.reshape(shape), new, old)

    result = _ParticleState(
        center_logits=select(proposal.center_logits, state.center_logits),
        log_coefficients_of_variation=select(
            proposal.log_coefficients_of_variation, state.log_coefficients_of_variation
        ),
        weight_logits=select(proposal.weight_logits, state.weight_logits),
        log_likelihood=select(proposal.log_likelihood, state.log_likelihood),
        log_transformed_prior=select(
            proposal.log_transformed_prior, state.log_transformed_prior
        ),
    )
    return result, float(accepted.to(torch.float64).mean())


def _adapt_step_scales(
    center_step: float,
    log_cv_step: float,
    weight_step: float,
    *,
    acceptance: float,
    settings: SMCSettings,
) -> tuple[float, float, float]:
    multiplier = math.exp(
        settings.adaptation_gain
        * (float(acceptance) - settings.adaptation_target_acceptance)
    )

    def update(value: float) -> float:
        return min(
            settings.maximum_step_scale,
            max(settings.minimum_step_scale, value * multiplier),
        )

    return update(center_step), update(log_cv_step), update(weight_step)


def _state_parameter_matrix(state: _ParticleState) -> torch.Tensor:
    parameters = (
        state.center_logits,
        state.log_coefficients_of_variation,
        state.weight_logits,
    )
    return torch.cat(parameters, dim=-1)


def _duplicate_criterion_met(
    diagnostics: dict[str, Any], settings: SMCSettings
) -> bool:
    return (
        diagnostics["unique_sample_fraction"] >= settings.final_minimum_unique_fraction
        and diagnostics["kth_neighbour_zero_radius_fraction"]
        <= settings.final_maximum_k3_zero_radius_fraction
    )


def run_fixed_j_smc(
    measurement: torch.Tensor,
    *,
    component_count: int,
    spec: ContinuousNMRSpec,
    grid: NMRGrid | None = None,
    settings: SMCSettings = SMCSettings(),
    seed: int,
    device: torch.device | str = "cpu",
) -> FixedJSMCResult:
    """Run an evidence-estimating tempered SMC sampler for one fixed J."""

    start_time = time.perf_counter()
    device = torch.device(device)
    grid = (build_nmr_grid(spec) if grid is None else grid).to(
        device=device, dtype=torch.float64
    )
    measurement = measurement.to(device=device, dtype=torch.float64)
    if tuple(measurement.shape) != (spec.num_measurements,):
        raise ValueError(
            f"measurement must have shape [{spec.num_measurements}], got {tuple(measurement.shape)}."
        )
    generator = private_generator(seed, device=device)
    initial_mixture = sample_gamma_mixture_prior(
        component_count,
        settings.num_particles,
        spec=spec,
        grid=grid.to(device="cpu", dtype=torch.float64),
        generator=private_generator(seed),
    )
    state = _initial_state(
        initial_mixture,
        component_count=component_count,
        spec=spec,
        measurement=measurement,
        grid=grid,
        device=device,
    )
    log_weights = torch.full(
        (settings.num_particles,),
        -math.log(settings.num_particles),
        dtype=torch.float64,
        device=device,
    )
    beta = 0.0
    log_evidence = 0.0
    center_step = settings.center_logit_step
    log_cv_step = settings.log_cv_step_fraction * math.log(
        spec.coefficient_of_variation_maximum / spec.coefficient_of_variation_minimum
    )
    weight_step = settings.weight_logit_step
    adaptation_sweeps = math.floor(
        settings.rejuvenation_sweeps * settings.tempering_adaptation_fraction
    )
    trace: list[dict[str, Any]] = []
    resampling_count = 0
    for temperature_index in range(settings.maximum_temperatures):
        next_beta = choose_next_beta(
            beta,
            log_weights,
            state.log_likelihood,
            target_fraction=settings.target_conditional_ess_fraction,
            bisection_iterations=settings.beta_bisection_iterations,
        )
        delta = next_beta - beta
        unnormalized = log_weights + delta * state.log_likelihood
        log_increment = torch.logsumexp(unnormalized, dim=0)
        log_evidence += float(log_increment)
        log_weights = unnormalized - log_increment
        beta = next_beta
        ess_before_resampling = _effective_sample_size(log_weights)
        resampled = (
            ess_before_resampling
            < settings.resample_ess_fraction * settings.num_particles
        )
        if resampled:
            indices = systematic_resample(log_weights, generator=generator)
            state = state.index_select(indices)
            log_weights.fill_(-math.log(settings.num_particles))
            resampling_count += 1
        acceptances: list[float] = []
        rejuvenation_trace: list[dict[str, Any]] = []
        # Tune at the current temperature before accumulating fixed-scale moves.
        # Adapting only once after a whole stage reacts too slowly when the
        # likelihood contracts.  The remaining sweeps use a frozen proposal;
        # independent-run agreement is still needed to establish exploration.
        for sweep_index in range(settings.rejuvenation_sweeps):
            sweep_record = {
                "sweep": sweep_index + 1,
                "adapted_after_sweep": sweep_index < adaptation_sweeps,
                "center_logit_step": center_step,
                "log_cv_step": log_cv_step,
                "weight_logit_step": weight_step,
            }
            state, acceptance = _metropolis_sweep(
                state,
                beta=beta,
                component_count=component_count,
                measurement=measurement,
                spec=spec,
                grid=grid,
                center_step=center_step,
                log_cv_step=log_cv_step,
                weight_step=weight_step,
                generator=generator,
            )
            acceptances.append(acceptance)
            sweep_record["acceptance"] = acceptance
            rejuvenation_trace.append(sweep_record)
            if sweep_index < adaptation_sweeps:
                center_step, log_cv_step, weight_step = _adapt_step_scales(
                    center_step,
                    log_cv_step,
                    weight_step,
                    acceptance=acceptance,
                    settings=settings,
                )
        mean_acceptance = (
            float(sum(acceptances) / len(acceptances)) if acceptances else None
        )
        fixed_acceptances = acceptances[adaptation_sweeps:]
        trace.append(
            {
                "temperature_index": temperature_index,
                "beta": beta,
                "delta_beta": delta,
                "ess_before_resampling": ess_before_resampling,
                "resampled": resampled,
                "mean_acceptance": mean_acceptance,
                "adaptation_sweeps": adaptation_sweeps,
                "fixed_scale_sweeps": len(fixed_acceptances),
                "fixed_scale_mean_acceptance": (
                    float(sum(fixed_acceptances) / len(fixed_acceptances))
                    if fixed_acceptances
                    else None
                ),
                "mean_accepted_moves_per_particle": float(sum(acceptances)),
                "center_logit_step": center_step,
                "log_cv_step": log_cv_step,
                "weight_logit_step": weight_step,
                "rejuvenation_trace": rejuvenation_trace,
            }
        )
        if beta >= 1.0 - 8.0 * torch.finfo(torch.float64).eps:
            beta = 1.0
            break
    else:
        raise RuntimeError(
            f"SMC did not reach beta=1 within {settings.maximum_temperatures} temperatures."
        )

    # The evidence estimate is complete at beta=1.  Convert the weighted cloud
    # to equal weights, then use only beta=1 Metropolis kernels; proposal-scale
    # adaptation changes efficiency but neither the target density nor log_evidence.
    indices = systematic_resample(log_weights, generator=generator)
    state = state.index_select(indices)
    log_weights.fill_(-math.log(settings.num_particles))
    post_resampling_duplicate_diagnostics = exact_duplicate_diagnostics(
        _state_parameter_matrix(state), nearest_k=3
    )
    final_acceptances: list[float] = []
    final_trace: list[dict[str, Any]] = []
    final_histogram_diagnostics: dict[str, Any] | None = None
    minimum_final_sweeps = max(
        settings.final_rejuvenation_sweeps,
        settings.final_acceptance_window,
    )
    for final_sweep in range(1, settings.maximum_final_rejuvenation_sweeps + 1):
        state, acceptance = _metropolis_sweep(
            state,
            beta=1.0,
            component_count=component_count,
            measurement=measurement,
            spec=spec,
            grid=grid,
            center_step=center_step,
            log_cv_step=log_cv_step,
            weight_step=weight_step,
            generator=generator,
        )
        final_acceptances.append(acceptance)
        center_step, log_cv_step, weight_step = _adapt_step_scales(
            center_step,
            log_cv_step,
            weight_step,
            acceptance=acceptance,
            settings=settings,
        )
        trace_record: dict[str, Any] = {
            "sweep": final_sweep,
            "acceptance": acceptance,
            "center_logit_step": center_step,
            "log_cv_step": log_cv_step,
            "weight_logit_step": weight_step,
        }
        final_trace.append(trace_record)
        should_check = final_sweep >= minimum_final_sweeps and (
            final_sweep == minimum_final_sweeps
            or (final_sweep - minimum_final_sweeps) % settings.final_diagnostic_interval
            == 0
            or final_sweep == settings.maximum_final_rejuvenation_sweeps
        )
        if not should_check:
            continue
        recent_acceptance = sum(
            final_acceptances[-settings.final_acceptance_window :]
        ) / min(len(final_acceptances), settings.final_acceptance_window)
        state_duplicate_diagnostics = exact_duplicate_diagnostics(
            _state_parameter_matrix(state), nearest_k=3
        )
        trace_record["recent_mean_acceptance"] = float(recent_acceptance)
        trace_record["state_duplicate_diagnostics"] = state_duplicate_diagnostics
        if (
            recent_acceptance < settings.final_minimum_mean_acceptance
            or not _duplicate_criterion_met(state_duplicate_diagnostics, settings)
        ):
            continue
        _, candidate_histograms, *_ = _evaluate_particle_state(
            state,
            component_count=component_count,
            measurement=measurement,
            spec=spec,
            grid=grid,
        )
        final_histogram_diagnostics = exact_duplicate_diagnostics(
            candidate_histograms, nearest_k=3
        )
        trace_record["histogram_duplicate_diagnostics"] = final_histogram_diagnostics
        if _duplicate_criterion_met(final_histogram_diagnostics, settings):
            break
    else:
        recent_acceptance = sum(
            final_acceptances[-settings.final_acceptance_window :]
        ) / min(len(final_acceptances), settings.final_acceptance_window)
        state_duplicate_diagnostics = exact_duplicate_diagnostics(
            _state_parameter_matrix(state), nearest_k=3
        )
        _, failed_histograms, *_ = _evaluate_particle_state(
            state,
            component_count=component_count,
            measurement=measurement,
            spec=spec,
            grid=grid,
        )
        final_histogram_diagnostics = exact_duplicate_diagnostics(
            failed_histograms, nearest_k=3
        )
        raise RuntimeError(
            "Final beta=1 SMC rejuvenation failed its mixing/duplicate gate "
            f"after {settings.maximum_final_rejuvenation_sweeps} sweeps for "
            f"J={component_count}: recent_mean_acceptance={recent_acceptance:.6g}, "
            "state_unique_fraction="
            f"{state_duplicate_diagnostics['unique_sample_fraction']:.6g}, "
            "histogram_unique_fraction="
            f"{final_histogram_diagnostics['unique_sample_fraction']:.6g}, "
            "histogram_k3_zero_radius_fraction="
            f"{final_histogram_diagnostics['kth_neighbour_zero_radius_fraction']:.6g}."
        )
    state, histograms, centers, cvs, weights, retained = _evaluate_particle_state(
        state,
        component_count=component_count,
        measurement=measurement,
        spec=spec,
        grid=grid,
    )
    final_histogram_diagnostics = exact_duplicate_diagnostics(histograms, nearest_k=3)
    diagnostics = {
        "sampler_version": settings.sampler_version,
        "elapsed_seconds": time.perf_counter() - start_time,
        "component_count": component_count,
        "num_particles": settings.num_particles,
        "num_temperatures": len(trace),
        "resampling_count": resampling_count,
        "post_resampling_state_duplicate_diagnostics": (
            post_resampling_duplicate_diagnostics
        ),
        "final_mean_acceptance": (
            float(sum(final_acceptances) / len(final_acceptances))
            if final_acceptances
            else None
        ),
        "final_rejuvenation_sweeps_completed": len(final_acceptances),
        "final_recent_mean_acceptance": float(
            sum(final_acceptances[-settings.final_acceptance_window :])
            / min(len(final_acceptances), settings.final_acceptance_window)
        ),
        "final_reference_duplicate_diagnostics": final_histogram_diagnostics,
        "final_rejuvenation_trace": final_trace,
        "temperature_trace": trace,
    }
    return FixedJSMCResult(
        component_count=component_count,
        histograms=histograms.detach().cpu(),
        centers=centers.detach().cpu(),
        coefficients_of_variation=cvs.detach().cpu(),
        mixture_weights=weights.detach().cpu(),
        retained_masses=retained.detach().cpu(),
        log_evidence=log_evidence,
        diagnostics=diagnostics,
    )


def _draw_component_count_assignments(
    probabilities: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.multinomial(
        probabilities.to(dtype=torch.float64),
        num_samples=count,
        replacement=True,
        generator=generator,
    )


def combine_fixed_j_results(
    results: list[FixedJSMCResult],
    *,
    spec: ContinuousNMRSpec,
    output_samples: int,
    seed: int,
) -> ReferenceConditionResult:
    if (
        isinstance(output_samples, bool)
        or not isinstance(output_samples, int)
        or output_samples <= 3
    ):
        raise ValueError("output_samples must be an integer greater than three.")
    if [result.component_count for result in results] != list(
        range(spec.component_count_minimum, spec.component_count_maximum + 1)
    ):
        raise ValueError("results must contain ordered J=1,...,5 fixed-J results.")
    if any(int(result.histograms.shape[0]) < output_samples for result in results):
        raise ValueError(
            "Each fixed-J cloud must contain at least output_samples rows so the "
            "compiled reference never samples with replacement."
        )
    log_evidence = torch.tensor(
        [result.log_evidence for result in results], dtype=torch.float64
    )
    log_prior_j = torch.full_like(log_evidence, -math.log(len(results)))
    probabilities = torch.softmax(log_prior_j + log_evidence, dim=0)
    generator = private_generator(seed)
    assignments = _draw_component_count_assignments(
        probabilities, output_samples, generator=generator
    )
    maximum = spec.component_count_maximum
    num_bins = spec.num_spectrum_bins
    histograms = torch.empty(output_samples, num_bins, dtype=torch.float64)
    component_counts = torch.empty(output_samples, dtype=torch.int64)
    active_mask = torch.zeros(output_samples, maximum, dtype=torch.bool)
    centers = torch.zeros(output_samples, maximum, dtype=torch.float64)
    cvs = torch.zeros_like(centers)
    weights = torch.zeros_like(centers)
    for result_index, result in enumerate(results):
        output_indices = torch.nonzero(
            assignments == result_index, as_tuple=False
        ).flatten()
        count = int(output_indices.numel())
        if count == 0:
            continue
        available = int(result.histograms.shape[0])
        source_indices = torch.randperm(available, generator=generator)[:count]
        j = result.component_count
        histograms[output_indices] = result.histograms[source_indices]
        component_counts[output_indices] = j
        active_mask[output_indices, :j] = True
        centers[output_indices, :j] = result.centers[source_indices]
        cvs[output_indices, :j] = result.coefficients_of_variation[source_indices]
        weights[output_indices, :j] = result.mixture_weights[source_indices]
    reference_histograms = histograms.to(torch.float32)
    compiled_duplicate_diagnostics = exact_duplicate_diagnostics(
        reference_histograms, nearest_k=3
    )
    return ReferenceConditionResult(
        reference_histograms=reference_histograms,
        reference_component_counts=component_counts,
        reference_active_mask=active_mask,
        reference_centers=centers.to(torch.float32),
        reference_coefficients_of_variation=cvs.to(torch.float32),
        reference_mixture_weights=weights.to(torch.float32),
        posterior_component_count_probability=probabilities,
        log_evidence=log_evidence,
        diagnostics={
            "fixed_j": [result.diagnostics for result in results],
            "posterior_component_count_probability": probabilities.tolist(),
            "output_component_count_histogram": torch.bincount(
                assignments, minlength=len(results)
            ).tolist(),
            "compiled_reference_duplicate_diagnostics": (
                compiled_duplicate_diagnostics
            ),
        },
    )


def run_reference_for_condition(
    measurement: torch.Tensor,
    *,
    condition_id: int,
    spec: ContinuousNMRSpec,
    settings: SMCSettings = SMCSettings(),
    master_seed: int,
    run_id: int,
    device: torch.device | str = "cpu",
) -> ReferenceConditionResult:
    grid = build_nmr_grid(spec)
    results: list[FixedJSMCResult] = []
    for component_count in range(
        spec.component_count_minimum, spec.component_count_maximum + 1
    ):
        seed = keyed_seed(
            protocol_id=spec.protocol_id,
            master_seed=master_seed,
            split=f"reference_run_{run_id}",
            sample_id=condition_id,
            stream=f"smc_j{component_count}",
        )
        results.append(
            run_fixed_j_smc(
                measurement,
                component_count=component_count,
                spec=spec,
                grid=grid,
                settings=settings,
                seed=seed,
                device=device,
            )
        )
    mixture_seed = keyed_seed(
        protocol_id=spec.protocol_id,
        master_seed=master_seed,
        split=f"reference_run_{run_id}",
        sample_id=condition_id,
        stream="model_order_mixture",
    )
    return combine_fixed_j_results(
        results,
        spec=spec,
        output_samples=settings.output_samples,
        seed=mixture_seed,
    )


def build_reference_artifact(
    fixed_test_split: dict[str, Any],
    *,
    spec: ContinuousNMRSpec,
    settings: SMCSettings = SMCSettings(),
    master_seed: int,
    run_id: int,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if fixed_test_split.get("specification_hash") != spec.specification_hash:
        raise ValueError("Test split and requested SMC specification hashes differ.")
    measurements = fixed_test_split["measurement"]
    if measurements.ndim != 2 or measurements.shape[1] != spec.num_measurements:
        raise ValueError("Test split measurement tensor has the wrong shape.")
    condition_results: list[ReferenceConditionResult] = []
    diagnostics: list[dict[str, Any]] = []
    for condition_index in range(int(measurements.shape[0])):
        result = run_reference_for_condition(
            measurements[condition_index],
            condition_id=int(fixed_test_split["sample_id"][condition_index]),
            spec=spec,
            settings=settings,
            master_seed=master_seed,
            run_id=run_id,
            device=device,
        )
        condition_results.append(result)
        diagnostics.append(
            {
                "condition_index": condition_index,
                "condition_id": int(fixed_test_split["sample_id"][condition_index]),
                **result.diagnostics,
            }
        )

    def stack(name: str) -> torch.Tensor:
        return torch.stack(
            [getattr(result, name) for result in condition_results], dim=0
        )

    artifact: dict[str, Any] = {
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "smc_settings": settings.to_dict(),
        "reference_run_id": int(run_id),
        "reference_master_seed": int(master_seed),
        "condition_ids": fixed_test_split["sample_id"].clone(),
        "measurements": fixed_test_split["measurement"].clone(),
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
            result.diagnostics["compiled_reference_duplicate_diagnostics"]
            for result in condition_results
        ],
        "grid": fixed_test_split["grid"],
    }
    artifact["truth_histograms"] = fixed_test_split["histogram"].clone()
    artifact["truth_component_counts"] = fixed_test_split["component_count"].clone()
    return artifact, diagnostics


def diagnostics_to_json(diagnostics: list[dict[str, Any]]) -> str:
    return json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False)
