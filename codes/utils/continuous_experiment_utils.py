from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset

try:  # Direct old-style entrypoint execution puts ``codes`` on sys.path.
    from data import (
        ContinuousGammaMixtureDataset,
        ContinuousNMRSpec,
        InfiniteGammaMixtureBatchStream,
    )
except ImportError:  # A local data/ cache may also be an empty namespace package.
    from codes.data import (
        ContinuousGammaMixtureDataset,
        ContinuousNMRSpec,
        InfiniteGammaMixtureBatchStream,
    )


_SPEC_ARGUMENTS: tuple[tuple[str, type], ...] = (
    ("num_measurements", int),
    ("num_spectrum_bins", int),
    ("rate_minimum", float),
    ("rate_maximum", float),
    ("echo_time_minimum", float),
    ("echo_time_maximum", float),
    ("noise_standard_deviation", float),
    ("component_count_minimum", int),
    ("component_count_maximum", int),
    ("center_minimum", float),
    ("center_maximum", float),
    ("coefficient_of_variation_minimum", float),
    ("coefficient_of_variation_maximum", float),
    ("mixture_weight_concentration", float),
    ("minimum_component_retained_mass", float),
    ("maximum_prior_attempts", int),
)


def add_continuous_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the complete, explicit v1 simulator contract to an entrypoint."""

    for name, value_type in _SPEC_ARGUMENTS:
        parser.add_argument(f"--{name}", type=value_type, required=True)


def spec_from_args(args: argparse.Namespace) -> ContinuousNMRSpec:
    values = {name: getattr(args, name) for name, _ in _SPEC_ARGUMENTS}
    # These are fixed by the versioned simulator rather than CLI choices.
    return ContinuousNMRSpec(**values)


def validate_continuous_dataset_arguments(args: argparse.Namespace) -> None:
    """Validate through the authoritative immutable simulator specification."""

    spec_from_args(args)


def continuous_dataset_config_values(args: argparse.Namespace) -> dict[str, Any]:
    spec = spec_from_args(args)
    return {
        "protocol_id": spec.protocol_id,
        "specification_hash": spec.specification_hash,
        "specification": spec.to_dict(),
    }


def json_safe_dataset_metadata(dataset_or_family: Any) -> dict[str, Any]:
    metadata = getattr(dataset_or_family, "generation_metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError("Dataset object has no mapping-valued generation_metadata.")
    return dict(metadata)


class _ContinuousView(Dataset):
    def __init__(self, family: "ContinuousDatasetFamily") -> None:
        self.family_dataset = family
        self._dataset = family.dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def set_epoch(self, epoch: int) -> None:
        self._dataset.set_epoch(epoch)

    @property
    def echo_times(self) -> torch.Tensor:
        return self._dataset.grid.echo_times

    @property
    def log_rate_edges(self) -> torch.Tensor:
        return self._dataset.grid.log_rate_edges

    @property
    def rate_centers(self) -> torch.Tensor:
        return self._dataset.grid.rate_centers

    @property
    def kernel(self) -> torch.Tensor:
        return self._dataset.grid.kernel

    @property
    def generation_metadata(self) -> dict[str, Any]:
        return self.family_dataset.generation_metadata

    @property
    def validation_report(self) -> dict[str, Any]:
        return self.family_dataset.validation_report


class _PosteriorWeightedTrainingView(_ContinuousView):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._dataset[index]
        return item["histogram"], item["measurement"], torch.tensor(1.0)


class _SingleTargetView(_ContinuousView):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self._dataset[index]
        return item["histogram"], item["measurement"]


class _FamilyEvaluationView(_ContinuousView):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self._dataset[index]
        sample_id = item["sample_id"].to(torch.int64)
        histogram = item["histogram"]
        return {
            "measurement": item["measurement"],
            "reference_histograms": histogram.unsqueeze(0),
            "posterior_weights": torch.ones(1, dtype=histogram.dtype),
            "target_histogram": histogram,
            "sample_id": sample_id,
            "condition_id": sample_id,
            "family_id": sample_id,
            "true_mode_id": torch.tensor(0, dtype=torch.int64),
        }


class ContinuousDatasetFamily:
    """Compatibility boundary for old entrypoint variable names, not a codebook."""

    def __init__(self, dataset: ContinuousGammaMixtureDataset) -> None:
        self.dataset = dataset
        self.num_families = len(dataset)
        spec = dataset.spec
        self.generation_metadata = {
            "protocol_id": spec.protocol_id,
            "specification_hash": spec.specification_hash,
            "specification": spec.to_dict(),
            "num_samples": len(dataset),
            "master_seed": dataset.master_seed,
            "split": dataset.split,
            "refresh_each_epoch": dataset.refresh_each_epoch,
        }
        self.validation_report = {
            "status": "validated_by_spec",
            "num_samples": len(dataset),
            "specification_hash": spec.specification_hash,
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    @property
    def echo_times(self) -> torch.Tensor:
        return self.dataset.grid.echo_times

    @property
    def log_rate_edges(self) -> torch.Tensor:
        return self.dataset.grid.log_rate_edges

    @property
    def rate_centers(self) -> torch.Tensor:
        return self.dataset.grid.rate_centers

    @property
    def kernel(self) -> torch.Tensor:
        return self.dataset.grid.kernel

    def as_posterior_weighted_training_dataset(self) -> _PosteriorWeightedTrainingView:
        return _PosteriorWeightedTrainingView(self)

    def as_single_target_dataset(self) -> _SingleTargetView:
        return _SingleTargetView(self)

    def as_family_evaluation_dataset(self) -> _FamilyEvaluationView:
        return _FamilyEvaluationView(self)


class _ContinuousTrainingStreamView(IterableDataset):
    def __init__(self, family: "ContinuousTrainingStreamFamily") -> None:
        self.family_stream = family
        self._stream = family.stream

    @property
    def echo_times(self) -> torch.Tensor:
        return self._stream.grid.echo_times

    @property
    def log_rate_edges(self) -> torch.Tensor:
        return self._stream.grid.log_rate_edges

    @property
    def rate_centers(self) -> torch.Tensor:
        return self._stream.grid.rate_centers

    @property
    def kernel(self) -> torch.Tensor:
        return self._stream.grid.kernel

    @property
    def generation_metadata(self) -> dict[str, Any]:
        return self.family_stream.generation_metadata


class _PosteriorWeightedTrainingStreamView(_ContinuousTrainingStreamView):
    def __iter__(self):
        for batch in self._stream:
            histograms = batch["histogram"]
            yield (
                histograms,
                batch["measurement"],
                torch.ones(histograms.shape[0], dtype=histograms.dtype),
            )


class _SingleTargetTrainingStreamView(_ContinuousTrainingStreamView):
    def __iter__(self):
        for batch in self._stream:
            yield batch["histogram"], batch["measurement"]


class ContinuousTrainingStreamFamily:
    """Old-style training views over a stateless infinite pair-ID stream."""

    def __init__(self, stream: InfiniteGammaMixtureBatchStream) -> None:
        self.stream = stream
        spec = stream.spec
        self.generation_metadata = {
            "protocol_id": spec.protocol_id,
            "specification_hash": spec.specification_hash,
            "specification": spec.to_dict(),
            "master_seed": stream.master_seed,
            "split": stream.split,
            "infinite_on_the_fly": True,
            "start_iteration": stream.start_iteration,
            "first_global_pair_id": stream.first_global_pair_id,
            "local_batch_size": stream.local_batch_size,
            "gradient_accumulation_batches": (
                stream.gradient_accumulation_batches
            ),
            "rank": stream.rank,
            "world_size": stream.world_size,
            "effective_global_batch_size": stream.effective_global_batch_size,
        }

    @property
    def echo_times(self) -> torch.Tensor:
        return self.stream.grid.echo_times

    @property
    def log_rate_edges(self) -> torch.Tensor:
        return self.stream.grid.log_rate_edges

    @property
    def rate_centers(self) -> torch.Tensor:
        return self.stream.grid.rate_centers

    @property
    def kernel(self) -> torch.Tensor:
        return self.stream.grid.kernel

    def as_posterior_weighted_training_stream(
        self,
    ) -> _PosteriorWeightedTrainingStreamView:
        return _PosteriorWeightedTrainingStreamView(self)

    def as_single_target_stream(self) -> _SingleTargetTrainingStreamView:
        return _SingleTargetTrainingStreamView(self)


def make_continuous_dataset_family(
    args: argparse.Namespace,
    *,
    num_samples: int,
    seed: int,
    split: str,
    refresh_each_epoch: bool,
) -> ContinuousDatasetFamily:
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63).")
    validate_continuous_dataset_arguments(args)
    return ContinuousDatasetFamily(
        ContinuousGammaMixtureDataset(
            num_samples=num_samples,
            spec=spec_from_args(args),
            master_seed=seed,
            split=split,
            refresh_each_epoch=refresh_each_epoch,
            include_metadata=True,
        )
    )


def make_continuous_training_stream(
    args: argparse.Namespace,
    *,
    start_iteration: int,
    local_batch_size: int,
    gradient_accumulation_batches: int,
    rank: int,
    world_size: int,
    seed: int,
    split: str = "train",
) -> ContinuousTrainingStreamFamily:
    """Build the rank-local, iteration-addressed on-the-fly training stream."""

    validate_continuous_dataset_arguments(args)
    return ContinuousTrainingStreamFamily(
        InfiniteGammaMixtureBatchStream(
            spec=spec_from_args(args),
            master_seed=seed,
            split=split,
            start_iteration=start_iteration,
            local_batch_size=local_batch_size,
            gradient_accumulation_batches=gradient_accumulation_batches,
            rank=rank,
            world_size=world_size,
            include_metadata=False,
        )
    )


__all__ = [
    "ContinuousDatasetFamily",
    "ContinuousTrainingStreamFamily",
    "add_continuous_dataset_arguments",
    "continuous_dataset_config_values",
    "json_safe_dataset_metadata",
    "make_continuous_dataset_family",
    "make_continuous_training_stream",
    "spec_from_args",
    "validate_continuous_dataset_arguments",
]
