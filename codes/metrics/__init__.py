"""NMR-specific posterior-sample metrics.

The package works directly on normalized relaxation-spectrum histograms.  It
does not use an image feature extractor and has no dependency on the legacy
``multi-mode`` project.
"""

from .distances import (
    DISTANCE_PROTOCOL_VERSION,
    GROUND_METRICS,
    canonical_metric_name,
    histograms_to_quantiles,
    pairwise_distances,
    validate_log_rate_edges,
)
from .energy_score import (
    ENERGY_SCORE_PROTOCOL_VERSION,
    compute_energy_score_report,
    histogram_l2_energy_score,
)
from .kernel_score import (
    KERNEL_SCORE_PROTOCOL_VERSION,
    compute_kernel_score_report,
    histogram_gaussian_kernel_score,
)
from .physics import (
    DEFAULT_RATE_WINDOWS,
    credible_interval_calibration,
    derived_quantities,
    forward_reduced_chi_square,
    forward_residual_diagnostics,
    rate_window_fractions,
    summarize_derived_quantities,
    validate_nonnegative_spectra,
)
from .prdc import (
    PRDC_PROTOCOL_VERSION,
    compute_prdc,
    compute_prdc_from_distances,
    kth_neighbour_radii,
)
from .sample_sets import (
    METRIC_PROTOCOL_VERSION,
    compute_sample_set_metrics,
    deterministic_reference_splits,
    directional_distances_from_cross,
    energy_distance_from_matrices,
    evaluate_grouped_samples,
    sample_set_diagnostics,
)


__all__ = [
    "DEFAULT_RATE_WINDOWS",
    "DISTANCE_PROTOCOL_VERSION",
    "ENERGY_SCORE_PROTOCOL_VERSION",
    "KERNEL_SCORE_PROTOCOL_VERSION",
    "GROUND_METRICS",
    "METRIC_PROTOCOL_VERSION",
    "PRDC_PROTOCOL_VERSION",
    "canonical_metric_name",
    "compute_energy_score_report",
    "compute_kernel_score_report",
    "compute_prdc",
    "compute_prdc_from_distances",
    "compute_sample_set_metrics",
    "credible_interval_calibration",
    "derived_quantities",
    "deterministic_reference_splits",
    "directional_distances_from_cross",
    "energy_distance_from_matrices",
    "evaluate_grouped_samples",
    "forward_reduced_chi_square",
    "forward_residual_diagnostics",
    "histograms_to_quantiles",
    "histogram_l2_energy_score",
    "histogram_gaussian_kernel_score",
    "kth_neighbour_radii",
    "pairwise_distances",
    "rate_window_fractions",
    "sample_set_diagnostics",
    "summarize_derived_quantities",
    "validate_nonnegative_spectra",
    "validate_log_rate_edges",
]
