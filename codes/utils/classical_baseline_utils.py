"""Classical non-negative inverse-Laplace baselines.

The implementations in this module follow the public multi-component T2
toolbox of Canales-Rodriguez et al. for the parts that are relevant to this
benchmark:

* ``nnls`` is unregularized non-negative least squares;
* ``x2_d2`` uses the toolbox's X2 rule (a 2 percent increase over the
  unregularized residual sum of squares) with its square Neumann-boundary
  second-difference penalty;
* ``bayesreg_i`` uses the published BayesReg evidence objective with the
  identity penalty.

``discrepancy_d2`` is also provided as an explicitly named, known-noise
alternative.  It is not silently substituted for X2, because the two rules
answer different questions.

Every solver returns both its raw NNLS amplitude estimate and a unit-mass copy.
The raw estimate is the faithful forward-model solution.  The normalized copy
is used by the benchmark's distribution metrics, whose inputs are probability
histograms.  Keeping both prevents post-hoc normalization from hiding a poor
amplitude or forward fit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import brentq, minimize_scalar, nnls as scipy_nnls
from scipy.special import erf


ClassicalMethod = Literal[
    "nnls",
    "x2_d2",
    "discrepancy_d2",
    "bayesreg_i",
]
BayesBetaSource = Literal["official_nnls_residual", "known_noise"]

CLASSICAL_METHODS: tuple[str, ...] = (
    "nnls",
    "x2_d2",
    "discrepancy_d2",
    "bayesreg_i",
)
BAYES_BETA_SOURCES: tuple[str, ...] = (
    "official_nnls_residual",
    "known_noise",
)
CLASSICAL_BASELINE_PROTOCOL_VERSION = "classical_nnls_nmr_v1"
CLASSICAL_BASELINE_REFERENCE_IMPLEMENTATION = (
    "https://github.com/ejcanalesr/multicomponent-T2-toolbox"
)
CLASSICAL_BASELINE_REFERENCE_PAPERS: tuple[str, ...] = (
    "https://doi.org/10.1016/j.media.2021.101959",
    "https://doi.org/10.1016/j.neuroimage.2021.118582",
)


@dataclass(frozen=True)
class ClassicalBaselineResult:
    """One condition's classical reconstruction and solver diagnostics."""

    method: str
    raw_spectrum: np.ndarray
    spectrum: np.ndarray
    regularization_parameter: float
    raw_mass: float
    raw_forward_sse: float
    normalized_forward_sse: float
    unregularized_forward_sse: float
    target_forward_sse: float | None
    achieved_residual_ratio: float | None
    beta: float | None
    objective_value: float | None
    objective_evaluations: int
    optimizer_success: bool
    selection_status: str


def _as_finite_float64_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty rank-two array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array)


def _as_finite_float64_vector(
    name: str,
    value: np.ndarray,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a nonempty rank-one array.")
    if expected_size is not None and int(array.size) != int(expected_size):
        raise ValueError(
            f"{name} must contain {expected_size} values, got {array.size}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array)


def make_neumann_second_difference(num_bins: int) -> np.ndarray:
    """Return the toolbox's square D2 matrix with Neumann boundaries.

    The first and last rows are ``[1,-1]`` and ``[-1,1]``; interior rows are
    ``[-1,2,-1]``.  This is intentionally not the shorter ``(N-2) x N``
    second-difference operator, because using the public toolbox convention is
    important for a reproducible baseline.
    """

    if isinstance(num_bins, bool) or not isinstance(num_bins, int):
        raise TypeError("num_bins must be an integer.")
    if num_bins < 2:
        raise ValueError("num_bins must be at least two.")
    penalty = np.diag(np.full(num_bins, 2.0, dtype=np.float64))
    off_diagonal = np.full(num_bins - 1, -1.0, dtype=np.float64)
    penalty += np.diag(off_diagonal, k=1)
    penalty += np.diag(off_diagonal, k=-1)
    penalty[0, 0] = 1.0
    penalty[-1, -1] = 1.0
    return penalty


def _nnls(
    design: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    if max_iterations <= 0:
        max_iterations = 10 * int(design.shape[1])
    solution, residual_norm = scipy_nnls(
        design,
        target,
        maxiter=int(max_iterations),
    )
    if not np.isfinite(solution).all() or not math.isfinite(float(residual_norm)):
        raise RuntimeError("NNLS produced a non-finite solution.")
    # scipy.nnls guarantees this up to its numerical tolerance.  Removing tiny
    # signed round-off here makes the serialized non-negativity contract exact.
    solution = np.maximum(solution, 0.0)
    return solution, float(residual_norm) ** 2


def _regularized_nnls(
    kernel: np.ndarray,
    measurement: np.ndarray,
    penalty: np.ndarray,
    regularization_parameter: float,
    *,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    regularization_parameter = float(regularization_parameter)
    if not math.isfinite(regularization_parameter) or regularization_parameter < 0:
        raise ValueError("regularization_parameter must be finite and nonnegative.")
    if regularization_parameter == 0.0:
        return _nnls(kernel, measurement, max_iterations=max_iterations)
    augmented_design = np.concatenate(
        (kernel, math.sqrt(regularization_parameter) * penalty), axis=0
    )
    augmented_target = np.concatenate(
        (measurement, np.zeros(penalty.shape[0], dtype=np.float64))
    )
    solution, _ = _nnls(
        augmented_design,
        augmented_target,
        max_iterations=max_iterations,
    )
    residual = kernel @ solution - measurement
    return solution, float(residual @ residual)


def _normalized_copy(raw_spectrum: np.ndarray) -> tuple[np.ndarray, float]:
    mass = float(raw_spectrum.sum())
    scale = max(1.0, float(np.linalg.norm(raw_spectrum, ord=1)))
    if not math.isfinite(mass) or mass <= np.finfo(np.float64).eps * scale:
        raise RuntimeError(
            "The classical solver returned zero spectrum mass; a probability "
            "histogram cannot be formed without inventing a reconstruction."
        )
    normalized = raw_spectrum / mass
    return normalized, mass


def _make_result(
    *,
    method: str,
    kernel: np.ndarray,
    measurement: np.ndarray,
    raw_spectrum: np.ndarray,
    regularization_parameter: float,
    unregularized_forward_sse: float,
    target_forward_sse: float | None,
    beta: float | None,
    objective_value: float | None,
    objective_evaluations: int,
    optimizer_success: bool,
    selection_status: str,
) -> ClassicalBaselineResult:
    spectrum, raw_mass = _normalized_copy(raw_spectrum)
    raw_residual = kernel @ raw_spectrum - measurement
    normalized_residual = kernel @ spectrum - measurement
    raw_sse = float(raw_residual @ raw_residual)
    normalized_sse = float(normalized_residual @ normalized_residual)
    ratio: float | None
    if unregularized_forward_sse > np.finfo(np.float64).tiny:
        ratio = raw_sse / float(unregularized_forward_sse)
    else:
        ratio = None
    return ClassicalBaselineResult(
        method=method,
        raw_spectrum=raw_spectrum,
        spectrum=spectrum,
        regularization_parameter=float(regularization_parameter),
        raw_mass=raw_mass,
        raw_forward_sse=raw_sse,
        normalized_forward_sse=normalized_sse,
        unregularized_forward_sse=float(unregularized_forward_sse),
        target_forward_sse=(
            None if target_forward_sse is None else float(target_forward_sse)
        ),
        achieved_residual_ratio=ratio,
        beta=None if beta is None else float(beta),
        objective_value=(None if objective_value is None else float(objective_value)),
        objective_evaluations=int(objective_evaluations),
        optimizer_success=bool(optimizer_success),
        selection_status=str(selection_status),
    )


def _solve_x2_d2(
    kernel: np.ndarray,
    measurement: np.ndarray,
    base_spectrum: np.ndarray,
    base_sse: float,
    *,
    x2_factor: float,
    regularization_maximum: float,
    optimizer_tolerance: float,
    optimizer_max_iterations: int,
    nnls_max_iterations: int,
) -> ClassicalBaselineResult:
    if not math.isfinite(x2_factor) or x2_factor <= 1.0:
        raise ValueError("x2_factor must be finite and greater than one.")
    penalty = make_neumann_second_difference(int(kernel.shape[1]))
    # This scale-aware floor only handles the mathematically undefined 0/0
    # case.  The official X2 objective is used whenever the base SSE is
    # numerically resolvable.
    sse_floor = np.finfo(np.float64).eps * max(1.0, float(measurement @ measurement))
    if base_sse <= sse_floor:
        return _make_result(
            method="x2_d2",
            kernel=kernel,
            measurement=measurement,
            raw_spectrum=base_spectrum,
            regularization_parameter=0.0,
            unregularized_forward_sse=base_sse,
            target_forward_sse=x2_factor * base_sse,
            beta=None,
            objective_value=0.0,
            objective_evaluations=0,
            optimizer_success=False,
            selection_status="zero_unregularized_residual_x2_undefined",
        )

    target_sse = float(x2_factor) * base_sse

    def objective(value: float) -> float:
        _, residual_sse = _regularized_nnls(
            kernel,
            measurement,
            penalty,
            value,
            max_iterations=nnls_max_iterations,
        )
        return abs(residual_sse - target_sse) / base_sse

    optimum = minimize_scalar(
        objective,
        bounds=(0.0, float(regularization_maximum)),
        method="bounded",
        options={
            "xatol": float(optimizer_tolerance),
            "maxiter": int(optimizer_max_iterations),
        },
    )
    regularization_parameter = float(optimum.x)
    raw_spectrum, _ = _regularized_nnls(
        kernel,
        measurement,
        penalty,
        regularization_parameter,
        max_iterations=nnls_max_iterations,
    )
    return _make_result(
        method="x2_d2",
        kernel=kernel,
        measurement=measurement,
        raw_spectrum=raw_spectrum,
        regularization_parameter=regularization_parameter,
        unregularized_forward_sse=base_sse,
        target_forward_sse=target_sse,
        beta=None,
        objective_value=float(optimum.fun),
        objective_evaluations=int(optimum.nfev),
        optimizer_success=bool(optimum.success),
        selection_status=(
            "x2_target_minimized" if optimum.success else "x2_optimizer_failed"
        ),
    )


def _solve_discrepancy_d2(
    kernel: np.ndarray,
    measurement: np.ndarray,
    base_spectrum: np.ndarray,
    base_sse: float,
    *,
    noise_standard_deviation: float,
    regularization_maximum: float,
    optimizer_tolerance: float,
    optimizer_max_iterations: int,
    nnls_max_iterations: int,
) -> ClassicalBaselineResult:
    penalty = make_neumann_second_difference(int(kernel.shape[1]))
    target_sse = int(kernel.shape[0]) * float(noise_standard_deviation) ** 2
    if base_sse >= target_sse:
        return _make_result(
            method="discrepancy_d2",
            kernel=kernel,
            measurement=measurement,
            raw_spectrum=base_spectrum,
            regularization_parameter=0.0,
            unregularized_forward_sse=base_sse,
            target_forward_sse=target_sse,
            beta=None,
            objective_value=abs(base_sse - target_sse),
            objective_evaluations=0,
            optimizer_success=True,
            selection_status="unregularized_residual_already_above_discrepancy",
        )

    _, maximum_sse = _regularized_nnls(
        kernel,
        measurement,
        penalty,
        regularization_maximum,
        max_iterations=nnls_max_iterations,
    )
    if maximum_sse < target_sse:
        raw_spectrum, _ = _regularized_nnls(
            kernel,
            measurement,
            penalty,
            regularization_maximum,
            max_iterations=nnls_max_iterations,
        )
        return _make_result(
            method="discrepancy_d2",
            kernel=kernel,
            measurement=measurement,
            raw_spectrum=raw_spectrum,
            regularization_parameter=regularization_maximum,
            unregularized_forward_sse=base_sse,
            target_forward_sse=target_sse,
            beta=None,
            objective_value=target_sse - maximum_sse,
            objective_evaluations=1,
            optimizer_success=False,
            selection_status="discrepancy_not_bracketed_at_lambda_maximum",
        )

    objective_evaluations = 0

    def root_function(value: float) -> float:
        nonlocal objective_evaluations
        objective_evaluations += 1
        _, residual_sse = _regularized_nnls(
            kernel,
            measurement,
            penalty,
            value,
            max_iterations=nnls_max_iterations,
        )
        return residual_sse - target_sse

    regularization_parameter = brentq(
        root_function,
        0.0,
        float(regularization_maximum),
        xtol=float(optimizer_tolerance),
        maxiter=int(optimizer_max_iterations),
    )
    raw_spectrum, residual_sse = _regularized_nnls(
        kernel,
        measurement,
        penalty,
        regularization_parameter,
        max_iterations=nnls_max_iterations,
    )
    return _make_result(
        method="discrepancy_d2",
        kernel=kernel,
        measurement=measurement,
        raw_spectrum=raw_spectrum,
        regularization_parameter=regularization_parameter,
        unregularized_forward_sse=base_sse,
        target_forward_sse=target_sse,
        beta=None,
        objective_value=abs(residual_sse - target_sse),
        objective_evaluations=objective_evaluations,
        optimizer_success=True,
        selection_status="known_noise_discrepancy_root",
    )


def _official_bayesreg_beta(
    base_spectrum: np.ndarray,
    base_sse: float,
    num_measurements: int,
) -> tuple[float, str]:
    # The official code counts every positive NNLS coefficient.  A tiny
    # relative threshold only prevents signed-zero numerical artifacts from
    # consuming degrees of freedom.
    coefficient_scale = max(1.0, float(base_spectrum.max(initial=0.0)))
    active_tolerance = 32.0 * np.finfo(np.float64).eps * coefficient_scale
    active_count = int(np.count_nonzero(base_spectrum > active_tolerance))
    degrees_of_freedom = max(int(num_measurements) - active_count, 1)
    variance = float(base_sse) / float(degrees_of_freedom)
    variance_floor = np.finfo(np.float64).eps
    if variance <= variance_floor:
        variance = variance_floor
        status = "official_nnls_residual_beta_variance_floored"
    else:
        status = "official_nnls_residual_beta"
    return 1.0 / variance, status


def _solve_bayesreg_i(
    kernel: np.ndarray,
    measurement: np.ndarray,
    base_spectrum: np.ndarray,
    base_sse: float,
    *,
    noise_standard_deviation: float,
    beta_source: BayesBetaSource,
    regularization_minimum: float,
    regularization_maximum: float,
    optimizer_tolerance: float,
    optimizer_max_iterations: int,
    nnls_max_iterations: int,
) -> ClassicalBaselineResult:
    num_measurements, num_bins = map(int, kernel.shape)
    identity = np.eye(num_bins, dtype=np.float64)
    if beta_source == "official_nnls_residual":
        beta, beta_status = _official_bayesreg_beta(
            base_spectrum,
            base_sse,
            num_measurements,
        )
    elif beta_source == "known_noise":
        beta = 1.0 / float(noise_standard_deviation) ** 2
        beta_status = "known_noise_beta"
    else:
        raise ValueError(
            f"beta_source must be one of {BAYES_BETA_SOURCES}, got {beta_source!r}."
        )

    gram = kernel.T @ kernel
    half_log_pi_over_two = 0.5 * num_bins * math.log(math.pi / 2.0)
    constant = (
        0.5 * num_measurements * math.log(2.0 * math.pi)
        - 0.5 * num_measurements * math.log(beta)
        + 0.5 * num_bins * math.log(math.pi)
    )
    tiny = np.finfo(np.float64).tiny

    def objective(value: float) -> float:
        spectrum, _ = _regularized_nnls(
            kernel,
            measurement,
            identity,
            value,
            max_iterations=nnls_max_iterations,
        )
        residual = kernel @ spectrum - measurement
        data_energy = 0.5 * float(residual @ residual)
        weight_energy = 0.5 * float(spectrum @ spectrum)
        precision = beta * (gram + float(value) * identity)
        upper = cholesky(
            precision,
            lower=False,
            overwrite_a=False,
            check_finite=False,
        )
        log_determinant_upper = float(np.log(np.diag(upper)).sum())
        orthant_terms = 1.0 + erf((upper @ spectrum) / math.sqrt(2.0))
        log_orthant_approximation = float(np.log(orthant_terms.clip(min=tiny)).sum())
        cost_one = (
            beta * data_energy
            + beta * float(value) * weight_energy
            + log_determinant_upper
            - half_log_pi_over_two
            - log_orthant_approximation
        )
        cost_two = constant - 0.5 * num_bins * math.log(2.0 * beta * float(value))
        result = cost_one + cost_two
        if not math.isfinite(result):
            return float("inf")
        return result

    optimum = minimize_scalar(
        objective,
        bounds=(float(regularization_minimum), float(regularization_maximum)),
        method="bounded",
        options={
            "xatol": float(optimizer_tolerance),
            "maxiter": int(optimizer_max_iterations),
        },
    )
    regularization_parameter = float(optimum.x)
    raw_spectrum, _ = _regularized_nnls(
        kernel,
        measurement,
        identity,
        regularization_parameter,
        max_iterations=nnls_max_iterations,
    )
    return _make_result(
        method="bayesreg_i",
        kernel=kernel,
        measurement=measurement,
        raw_spectrum=raw_spectrum,
        regularization_parameter=regularization_parameter,
        unregularized_forward_sse=base_sse,
        target_forward_sse=None,
        beta=beta,
        objective_value=float(optimum.fun),
        objective_evaluations=int(optimum.nfev),
        optimizer_success=bool(optimum.success),
        selection_status=(
            f"{beta_status};bayes_evidence_minimized"
            if optimum.success
            else f"{beta_status};bayes_optimizer_failed"
        ),
    )


def solve_classical_baseline(
    kernel: np.ndarray,
    measurement: np.ndarray,
    *,
    method: ClassicalMethod,
    noise_standard_deviation: float,
    x2_factor: float = 1.02,
    bayes_beta_source: BayesBetaSource = "official_nnls_residual",
    regularization_minimum: float = 1.0e-8,
    regularization_maximum: float | None = None,
    optimizer_tolerance: float = 1.0e-5,
    optimizer_max_iterations: int = 300,
    nnls_max_iterations: int = 0,
) -> ClassicalBaselineResult:
    """Solve one measurement condition using the requested classical method."""

    kernel = _as_finite_float64_matrix("kernel", kernel)
    measurement = _as_finite_float64_vector(
        "measurement", measurement, expected_size=int(kernel.shape[0])
    )
    if method not in CLASSICAL_METHODS:
        raise ValueError(f"method must be one of {CLASSICAL_METHODS}, got {method!r}.")
    if (
        isinstance(noise_standard_deviation, bool)
        or not isinstance(noise_standard_deviation, (int, float))
        or not math.isfinite(float(noise_standard_deviation))
        or float(noise_standard_deviation) <= 0.0
    ):
        raise ValueError("noise_standard_deviation must be finite and positive.")
    if optimizer_tolerance <= 0.0 or not math.isfinite(optimizer_tolerance):
        raise ValueError("optimizer_tolerance must be finite and positive.")
    if optimizer_max_iterations <= 0:
        raise ValueError("optimizer_max_iterations must be positive.")
    if nnls_max_iterations < 0:
        raise ValueError("nnls_max_iterations must be nonnegative.")
    if regularization_minimum <= 0.0 or not math.isfinite(regularization_minimum):
        raise ValueError("regularization_minimum must be finite and positive.")
    if regularization_maximum is None:
        regularization_maximum = 2.0 if method == "bayesreg_i" else 10.0
    if (
        not math.isfinite(float(regularization_maximum))
        or float(regularization_maximum) <= regularization_minimum
    ):
        raise ValueError(
            "regularization_maximum must be finite and exceed the minimum."
        )

    base_spectrum, base_sse = _nnls(
        kernel,
        measurement,
        max_iterations=nnls_max_iterations,
    )
    if method == "nnls":
        return _make_result(
            method=method,
            kernel=kernel,
            measurement=measurement,
            raw_spectrum=base_spectrum,
            regularization_parameter=0.0,
            unregularized_forward_sse=base_sse,
            target_forward_sse=None,
            beta=None,
            objective_value=None,
            objective_evaluations=1,
            optimizer_success=True,
            selection_status="unregularized_nnls",
        )
    if method == "x2_d2":
        return _solve_x2_d2(
            kernel,
            measurement,
            base_spectrum,
            base_sse,
            x2_factor=x2_factor,
            regularization_maximum=float(regularization_maximum),
            optimizer_tolerance=optimizer_tolerance,
            optimizer_max_iterations=optimizer_max_iterations,
            nnls_max_iterations=nnls_max_iterations,
        )
    if method == "discrepancy_d2":
        return _solve_discrepancy_d2(
            kernel,
            measurement,
            base_spectrum,
            base_sse,
            noise_standard_deviation=noise_standard_deviation,
            regularization_maximum=float(regularization_maximum),
            optimizer_tolerance=optimizer_tolerance,
            optimizer_max_iterations=optimizer_max_iterations,
            nnls_max_iterations=nnls_max_iterations,
        )
    return _solve_bayesreg_i(
        kernel,
        measurement,
        base_spectrum,
        base_sse,
        noise_standard_deviation=noise_standard_deviation,
        beta_source=bayes_beta_source,
        regularization_minimum=regularization_minimum,
        regularization_maximum=float(regularization_maximum),
        optimizer_tolerance=optimizer_tolerance,
        optimizer_max_iterations=optimizer_max_iterations,
        nnls_max_iterations=nnls_max_iterations,
    )


def solve_classical_baseline_batch(
    kernel: np.ndarray,
    measurements: np.ndarray,
    *,
    method: ClassicalMethod,
    noise_standard_deviation: float,
    num_workers: int = 1,
    **solver_options: object,
) -> list[ClassicalBaselineResult]:
    """Solve a batch while preserving input order.

    SciPy's NNLS kernels release the GIL, so a thread pool avoids repeatedly
    copying the shared forward matrix into worker processes.  ``num_workers=1``
    remains the deterministic low-overhead path used by unit tests.
    """

    kernel_array = _as_finite_float64_matrix("kernel", kernel)
    measurement_array = _as_finite_float64_matrix("measurements", measurements)
    if int(measurement_array.shape[1]) != int(kernel_array.shape[0]):
        raise ValueError("measurements must have shape [G,M] matching kernel.shape[0].")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be an integer.")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")

    def solve_one(measurement: np.ndarray) -> ClassicalBaselineResult:
        return solve_classical_baseline(
            kernel_array,
            measurement,
            method=method,
            noise_standard_deviation=noise_standard_deviation,
            **solver_options,
        )

    rows: Iterable[np.ndarray] = (
        measurement_array[index] for index in range(measurement_array.shape[0])
    )
    if num_workers == 1:
        return [solve_one(row) for row in rows]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(solve_one, rows))


__all__ = [
    "BAYES_BETA_SOURCES",
    "CLASSICAL_BASELINE_PROTOCOL_VERSION",
    "CLASSICAL_BASELINE_REFERENCE_IMPLEMENTATION",
    "CLASSICAL_BASELINE_REFERENCE_PAPERS",
    "CLASSICAL_METHODS",
    "ClassicalBaselineResult",
    "make_neumann_second_difference",
    "solve_classical_baseline",
    "solve_classical_baseline_batch",
]
