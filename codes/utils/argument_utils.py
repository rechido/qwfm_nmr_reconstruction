from __future__ import annotations

import argparse

from codes.data.spec import ContinuousNMRSpec
from codes.utils.continuous_experiment_utils import (
    spec_from_args as _shared_spec_from_args,
)


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, 1/0, or yes/no.")


def add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = ContinuousNMRSpec()
    parser.add_argument("--num_measurements", type=int, default=defaults.num_measurements)
    parser.add_argument("--num_spectrum_bins", type=int, default=defaults.num_spectrum_bins)
    parser.add_argument("--rate_minimum", type=float, default=defaults.rate_minimum)
    parser.add_argument("--rate_maximum", type=float, default=defaults.rate_maximum)
    parser.add_argument("--echo_time_minimum", type=float, default=defaults.echo_time_minimum)
    parser.add_argument("--echo_time_maximum", type=float, default=defaults.echo_time_maximum)
    parser.add_argument(
        "--noise_standard_deviation",
        type=float,
        default=defaults.noise_standard_deviation,
    )
    parser.add_argument(
        "--component_count_minimum", type=int, default=defaults.component_count_minimum
    )
    parser.add_argument(
        "--component_count_maximum", type=int, default=defaults.component_count_maximum
    )
    parser.add_argument("--center_minimum", type=float, default=defaults.center_minimum)
    parser.add_argument("--center_maximum", type=float, default=defaults.center_maximum)
    parser.add_argument(
        "--coefficient_of_variation_minimum",
        type=float,
        default=defaults.coefficient_of_variation_minimum,
    )
    parser.add_argument(
        "--coefficient_of_variation_maximum",
        type=float,
        default=defaults.coefficient_of_variation_maximum,
    )
    parser.add_argument(
        "--mixture_weight_concentration",
        type=float,
        default=defaults.mixture_weight_concentration,
    )
    parser.add_argument(
        "--minimum_component_retained_mass",
        type=float,
        default=defaults.minimum_component_retained_mass,
    )
    parser.add_argument(
        "--maximum_prior_attempts", type=int, default=defaults.maximum_prior_attempts
    )


def spec_from_args(args: argparse.Namespace) -> ContinuousNMRSpec:
    return _shared_spec_from_args(args)
