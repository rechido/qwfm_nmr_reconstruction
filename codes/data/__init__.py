"""Data generation and reference-posterior inference."""

from .grid import NMRGrid, build_nmr_grid
from .simulator import (
    ContinuousGammaMixtureDataset,
    GammaMixtureBatch,
    InfiniteGammaMixtureBatchStream,
    generate_fixed_split,
    sample_gamma_mixture_prior,
    simulate_from_key,
)
from .spec import ContinuousNMRSpec

__all__ = [
    "ContinuousGammaMixtureDataset",
    "ContinuousNMRSpec",
    "GammaMixtureBatch",
    "InfiniteGammaMixtureBatchStream",
    "NMRGrid",
    "build_nmr_grid",
    "generate_fixed_split",
    "sample_gamma_mixture_prior",
    "simulate_from_key",
]
