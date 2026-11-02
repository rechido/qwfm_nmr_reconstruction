from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any


PROTOCOL_ID = "continuous_variable_gamma_mixture_v1"


@dataclass(frozen=True)
class ContinuousNMRSpec:
    """Immutable simulator contract shared by training, SMC, and evaluation."""

    protocol_id: str = PROTOCOL_ID
    num_measurements: int = 32
    num_spectrum_bins: int = 128
    rate_minimum: float = 2.0
    rate_maximum: float = 250.0
    echo_time_minimum: float = 0.008
    echo_time_maximum: float = 0.512
    canonical_echo_count: int | None = None
    noise_standard_deviation: float = 0.005
    component_count_minimum: int = 1
    component_count_maximum: int = 5
    center_minimum: float = 4.0
    center_maximum: float = 160.0
    coefficient_of_variation_minimum: float = 0.04
    coefficient_of_variation_maximum: float = 0.25
    mixture_weight_concentration: float = 2.0
    minimum_component_retained_mass: float = 0.995
    maximum_prior_attempts: int = 100_000

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError("Unsupported continuous NMR protocol_id.")
        if self.canonical_echo_count is None and isinstance(self.num_measurements, int):
            object.__setattr__(self, "canonical_echo_count", max(64, self.num_measurements))
        integer_fields = (
            "num_measurements",
            "num_spectrum_bins",
            "canonical_echo_count",
            "component_count_minimum",
            "component_count_maximum",
            "maximum_prior_attempts",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_measurements not in (32, 64, 128):
            raise ValueError("Supported measurement schedules are M=32, M=64, and M=128.")
        if self.canonical_echo_count != max(64, self.num_measurements):
            raise ValueError("canonical_echo_count must equal max(64, num_measurements).")
        if self.num_spectrum_bins < 8 or self.num_spectrum_bins % 8:
            raise ValueError("num_spectrum_bins must be a positive multiple of eight.")
        if self.component_count_minimum != 1 or self.component_count_maximum != 5:
            raise ValueError("Version 1 fixes J to Uniform{1,...,5}.")
        real_fields = (
            "rate_minimum",
            "rate_maximum",
            "echo_time_minimum",
            "echo_time_maximum",
            "noise_standard_deviation",
            "center_minimum",
            "center_maximum",
            "coefficient_of_variation_minimum",
            "coefficient_of_variation_maximum",
            "mixture_weight_concentration",
            "minimum_component_retained_mass",
        )
        for name in real_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be real-valued.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 < self.rate_minimum < self.rate_maximum:
            raise ValueError("Require 0 < rate_minimum < rate_maximum.")
        if not 0.0 < self.echo_time_minimum < self.echo_time_maximum:
            raise ValueError("Require 0 < echo_time_minimum < echo_time_maximum.")
        if self.noise_standard_deviation <= 0.0:
            raise ValueError("noise_standard_deviation must be positive.")
        if not self.rate_minimum <= self.center_minimum < self.center_maximum <= self.rate_maximum:
            raise ValueError("Center bounds must lie inside the reconstruction interval.")
        if not 0.0 < self.coefficient_of_variation_minimum < self.coefficient_of_variation_maximum:
            raise ValueError("Coefficient-of-variation bounds must be positive and ordered.")
        if self.mixture_weight_concentration <= 0.0:
            raise ValueError("mixture_weight_concentration must be positive.")
        if not 0.0 < self.minimum_component_retained_mass <= 1.0:
            raise ValueError("minimum_component_retained_mass must lie in (0,1].")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def specification_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ContinuousNMRSpec":
        return cls(**values)
