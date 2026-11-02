from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset, default_collate, get_worker_info

from .grid import NMRGrid, build_nmr_grid
from .spec import ContinuousNMRSpec


_SEED_MODULUS = 2**63 - 1


def keyed_seed(
    *,
    protocol_id: str,
    master_seed: int,
    split: str,
    sample_id: int,
    stream: str,
    epoch: int | None = None,
    attempt: int | None = None,
) -> int:
    """Create a stable private RNG seed without touching global RNG state."""

    if master_seed < 0 or master_seed >= 2**63:
        raise ValueError("master_seed must lie in [0,2**63).")
    pieces: Iterable[object] = (
        protocol_id,
        master_seed,
        split,
        "fixed" if epoch is None else epoch,
        sample_id,
        stream,
        "none" if attempt is None else attempt,
    )
    payload = "|".join(str(piece) for piece in pieces).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value % _SEED_MODULUS


def private_generator(seed: int, *, device: str | torch.device = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


@dataclass(frozen=True)
class GammaMixtureBatch:
    component_counts: torch.Tensor
    active_mask: torch.Tensor
    centers: torch.Tensor
    coefficients_of_variation: torch.Tensor
    mixture_weights: torch.Tensor
    retained_masses: torch.Tensor
    histograms: torch.Tensor

    def __len__(self) -> int:
        return int(self.histograms.shape[0])

    def to(self, *args: Any, **kwargs: Any) -> "GammaMixtureBatch":
        return GammaMixtureBatch(
            **{
                name: getattr(self, name).to(*args, **kwargs)
                for name in self.__dataclass_fields__
            }
        )


def _draw_dirichlet(
    count: int,
    dimension: int,
    concentration: float,
    generator: torch.Generator,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    alpha = torch.full((count, dimension), concentration, dtype=dtype)
    draws = torch._standard_gamma(alpha, generator=generator)
    if not bool(torch.isfinite(draws).all()) or bool((draws <= 0.0).any()):
        raise RuntimeError("Gamma draws for Dirichlet weights were invalid.")
    return draws / draws.sum(dim=-1, keepdim=True)


def gamma_component_bin_masses(
    centers: torch.Tensor,
    coefficients_of_variation: torch.Tensor,
    rate_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unnormalized component bin masses and retained interval mass."""

    if centers.shape != coefficients_of_variation.shape or centers.ndim != 2:
        raise ValueError("centers and coefficients_of_variation must have shape [B,J].")
    if rate_edges.ndim != 1 or rate_edges.numel() < 3:
        raise ValueError("rate_edges must contain at least three ordered values.")
    if bool((centers <= 0.0).any()) or bool((coefficients_of_variation <= 0.0).any()):
        raise ValueError("Gamma centers and coefficients of variation must be positive.")
    shape = coefficients_of_variation.reciprocal().square()
    scale = centers * coefficients_of_variation.square()
    scaled_edges = rate_edges.to(device=centers.device, dtype=centers.dtype)[None, None, :]
    scaled_edges = scaled_edges / scale[:, :, None]
    cdf = torch.special.gammainc(shape[:, :, None], scaled_edges)
    masses = torch.diff(cdf, dim=-1).clamp_min(0.0)
    retained = masses.sum(dim=-1)
    return masses, retained


def gamma_mixture_histograms(
    centers: torch.Tensor,
    coefficients_of_variation: torch.Tensor,
    weights: torch.Tensor,
    rate_edges: torch.Tensor,
    *,
    minimum_retained_mass: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if centers.shape != coefficients_of_variation.shape or centers.shape != weights.shape:
        raise ValueError("centers, coefficients_of_variation, and weights must match.")
    if not torch.allclose(
        weights.sum(dim=-1),
        torch.ones(weights.shape[0], dtype=weights.dtype, device=weights.device),
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError("Mixture weights must sum to one.")
    component_masses, retained = gamma_component_bin_masses(
        centers, coefficients_of_variation, rate_edges
    )
    if minimum_retained_mass is not None and bool((retained < minimum_retained_mass).any()):
        raise ValueError("At least one component violates the retained-mass gate.")
    if bool((retained <= 0.0).any()) or not bool(torch.isfinite(retained).all()):
        raise RuntimeError("Gamma component retained mass is invalid.")
    normalized_components = component_masses / retained[:, :, None]
    histograms = (weights[:, :, None] * normalized_components).sum(dim=1)
    histograms = histograms.clamp_min(0.0)
    histograms = histograms / histograms.sum(dim=-1, keepdim=True)
    return histograms, retained


def sample_gamma_mixture_prior(
    component_count: int,
    count: int,
    *,
    spec: ContinuousNMRSpec,
    grid: NMRGrid,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> GammaMixtureBatch:
    """Draw exactly from the retained-mass-conditioned prior for fixed J."""

    if component_count < spec.component_count_minimum or component_count > spec.component_count_maximum:
        raise ValueError("component_count lies outside the configured prior.")
    if count <= 0:
        raise ValueError("count must be positive.")
    accepted_centers: list[torch.Tensor] = []
    accepted_cv: list[torch.Tensor] = []
    accepted_weights: list[torch.Tensor] = []
    accepted_retained: list[torch.Tensor] = []
    accepted_histograms: list[torch.Tensor] = []
    total = 0
    attempted = 0
    log_center_min = math.log(spec.center_minimum)
    log_center_max = math.log(spec.center_maximum)
    log_cv_min = math.log(spec.coefficient_of_variation_minimum)
    log_cv_max = math.log(spec.coefficient_of_variation_maximum)
    while total < count:
        # Gate acceptance is deliberately high for the fixed prior.  Drawing
        # only the missing count avoids a 32x cost when the on-the-fly Dataset
        # asks for one item, while still batching the large SMC initialization.
        proposal_count = min(count - total, spec.maximum_prior_attempts - attempted)
        if proposal_count <= 0:
            raise RuntimeError(
                f"Retained-mass prior gate failed to produce {count} draws after "
                f"{spec.maximum_prior_attempts} proposals."
            )
        attempted += proposal_count
        centers = torch.empty(proposal_count, component_count, dtype=dtype).uniform_(
            log_center_min, log_center_max, generator=generator
        ).exp_()
        cvs = torch.empty(proposal_count, component_count, dtype=dtype).uniform_(
            log_cv_min, log_cv_max, generator=generator
        ).exp_()
        weights = _draw_dirichlet(
            proposal_count,
            component_count,
            spec.mixture_weight_concentration,
            generator,
            dtype=dtype,
        )
        order = centers.argsort(dim=-1)
        centers = centers.gather(1, order)
        cvs = cvs.gather(1, order)
        weights = weights.gather(1, order)
        component_masses, retained = gamma_component_bin_masses(
            centers, cvs, grid.rate_edges.to(dtype=dtype)
        )
        valid = (retained >= spec.minimum_component_retained_mass).all(dim=-1)
        if not bool(valid.any()):
            continue
        centers = centers[valid]
        cvs = cvs[valid]
        weights = weights[valid]
        retained = retained[valid]
        component_masses = component_masses[valid] / retained[:, :, None]
        histograms = (weights[:, :, None] * component_masses).sum(dim=1)
        histograms = histograms / histograms.sum(dim=-1, keepdim=True)
        remaining = count - total
        keep = min(remaining, int(histograms.shape[0]))
        accepted_centers.append(centers[:keep])
        accepted_cv.append(cvs[:keep])
        accepted_weights.append(weights[:keep])
        accepted_retained.append(retained[:keep])
        accepted_histograms.append(histograms[:keep])
        total += keep
    centers = torch.cat(accepted_centers, dim=0)
    cvs = torch.cat(accepted_cv, dim=0)
    weights = torch.cat(accepted_weights, dim=0)
    retained = torch.cat(accepted_retained, dim=0)
    histograms = torch.cat(accepted_histograms, dim=0)
    maximum = spec.component_count_maximum
    active_mask = torch.arange(maximum)[None, :] < component_count

    def pad(values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(count, maximum, dtype=values.dtype)
        result[:, :component_count] = values
        return result

    return GammaMixtureBatch(
        component_counts=torch.full((count,), component_count, dtype=torch.int64),
        active_mask=active_mask.expand(count, -1).clone(),
        centers=pad(centers),
        coefficients_of_variation=pad(cvs),
        mixture_weights=pad(weights),
        retained_masses=pad(retained),
        histograms=histograms,
    )


def simulate_from_key(
    sample_id: int,
    *,
    spec: ContinuousNMRSpec,
    grid: NMRGrid,
    master_seed: int,
    split: str,
    epoch: int | None,
) -> dict[str, torch.Tensor]:
    structure_seed = keyed_seed(
        protocol_id=spec.protocol_id,
        master_seed=master_seed,
        split=split,
        epoch=epoch,
        sample_id=sample_id,
        stream="structure",
    )
    structure_generator = private_generator(structure_seed)
    component_count = int(
        torch.randint(
            spec.component_count_minimum,
            spec.component_count_maximum + 1,
            (1,),
            generator=structure_generator,
        ).item()
    )
    theta_seed = keyed_seed(
        protocol_id=spec.protocol_id,
        master_seed=master_seed,
        split=split,
        epoch=epoch,
        sample_id=sample_id,
        stream="theta",
    )
    mixture = sample_gamma_mixture_prior(
        component_count,
        1,
        spec=spec,
        grid=grid,
        generator=private_generator(theta_seed),
    )
    histogram = mixture.histograms[0]
    clean_measurement = grid.kernel @ histogram
    noise_seed = keyed_seed(
        protocol_id=spec.protocol_id,
        master_seed=master_seed,
        split=split,
        epoch=epoch,
        sample_id=sample_id,
        stream="noise",
    )
    noise = torch.randn(
        spec.num_measurements,
        generator=private_generator(noise_seed),
        dtype=torch.float64,
    ) * spec.noise_standard_deviation
    return {
        "sample_id": torch.tensor(sample_id, dtype=torch.int64),
        "histogram": histogram.to(torch.float32),
        "clean_measurement": clean_measurement.to(torch.float32),
        "noise": noise.to(torch.float32),
        "measurement": (clean_measurement + noise).to(torch.float32),
        "component_count": mixture.component_counts[0],
        "active_mask": mixture.active_mask[0],
        "centers": mixture.centers[0],
        "coefficients_of_variation": mixture.coefficients_of_variation[0],
        "mixture_weights": mixture.mixture_weights[0],
        "retained_masses": mixture.retained_masses[0],
    }


class ContinuousGammaMixtureDataset(Dataset[dict[str, torch.Tensor]]):
    """Stateless on-the-fly joint simulator keyed by epoch and sample index."""

    def __init__(
        self,
        *,
        num_samples: int,
        spec: ContinuousNMRSpec,
        master_seed: int,
        split: str,
        refresh_each_epoch: bool,
        include_metadata: bool = True,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not split or "|" in split:
            raise ValueError("split must be a nonempty seed-namespace token.")
        self.num_samples = int(num_samples)
        self.spec = spec
        self.grid = build_nmr_grid(spec)
        self.master_seed = int(master_seed)
        self.split = split
        self.refresh_each_epoch = bool(refresh_each_epoch)
        self.include_metadata = bool(include_metadata)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer.")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        item = simulate_from_key(
            index,
            spec=self.spec,
            grid=self.grid,
            master_seed=self.master_seed,
            split=self.split,
            epoch=self.epoch if self.refresh_each_epoch else None,
        )
        if self.include_metadata:
            return item
        return {
            "histogram": item["histogram"],
            "measurement": item["measurement"],
        }


class InfiniteGammaMixtureBatchStream(
    IterableDataset[dict[str, torch.Tensor]]
):
    """Infinite rank-local microbatches keyed only by global training-pair IDs.

    ``start_iteration`` is the number of completed optimizer updates.  For an
    optimizer iteration ``i``, accumulation slot ``a``, and distributed rank
    ``r``, this stream assigns the contiguous ID interval

    ``[iG + aWB + rB, iG + aWB + (r + 1)B)``,

    where ``B`` is the local microbatch size, ``W`` is the world size, and
    ``G = B * W * A`` for ``A`` gradient-accumulation slots.  Consequently,
    all ranks together consume exactly ``[iG, (i + 1)G)`` per optimizer
    update, independently of the local-microbatch/accumulation factorization.

    Each item yielded by this ``IterableDataset`` is already a complete local
    microbatch.  It must therefore be loaded with ``DataLoader(batch_size=None)``.
    The stream performs its own rank partitioning and must not be sharded a
    second time by a distributed data-loader wrapper.
    """

    def __init__(
        self,
        *,
        spec: ContinuousNMRSpec,
        master_seed: int,
        split: str,
        start_iteration: int,
        local_batch_size: int,
        gradient_accumulation_batches: int,
        rank: int,
        world_size: int,
        include_metadata: bool = True,
    ) -> None:
        integer_values = {
            "start_iteration": start_iteration,
            "local_batch_size": local_batch_size,
            "gradient_accumulation_batches": gradient_accumulation_batches,
            "rank": rank,
            "world_size": world_size,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if start_iteration < 0:
            raise ValueError("start_iteration must be nonnegative.")
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive.")
        if gradient_accumulation_batches <= 0:
            raise ValueError("gradient_accumulation_batches must be positive.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must lie in [0, world_size).")
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or master_seed < 0
            or master_seed >= 2**63
        ):
            raise ValueError("master_seed must be an integer in [0,2**63).")
        if not split or "|" in split:
            raise ValueError("split must be a nonempty seed-namespace token.")

        self.spec = spec
        self.grid = build_nmr_grid(spec)
        self.master_seed = master_seed
        self.split = split
        self.start_iteration = start_iteration
        self.local_batch_size = local_batch_size
        self.gradient_accumulation_batches = gradient_accumulation_batches
        self.rank = rank
        self.world_size = world_size
        self.include_metadata = bool(include_metadata)

    @property
    def effective_global_batch_size(self) -> int:
        return (
            self.local_batch_size
            * self.world_size
            * self.gradient_accumulation_batches
        )

    @property
    def global_microbatch_size(self) -> int:
        return self.local_batch_size * self.world_size

    @property
    def first_global_pair_id(self) -> int:
        return self.start_iteration * self.effective_global_batch_size

    def global_pair_ids_for_microbatch(
        self, microbatch_offset: int
    ) -> torch.Tensor:
        """Return this rank's IDs for a zero-based post-resume microbatch."""

        if (
            isinstance(microbatch_offset, bool)
            or not isinstance(microbatch_offset, int)
            or microbatch_offset < 0
        ):
            raise ValueError("microbatch_offset must be a nonnegative integer.")
        absolute_microbatch = (
            self.start_iteration * self.gradient_accumulation_batches
            + microbatch_offset
        )
        optimizer_iteration, accumulation_slot = divmod(
            absolute_microbatch, self.gradient_accumulation_batches
        )
        first = (
            optimizer_iteration * self.effective_global_batch_size
            + accumulation_slot * self.global_microbatch_size
            + self.rank * self.local_batch_size
        )
        return torch.arange(
            first,
            first + self.local_batch_size,
            dtype=torch.int64,
        )

    def _simulate_microbatch(self, microbatch_offset: int) -> dict[str, torch.Tensor]:
        pair_ids = self.global_pair_ids_for_microbatch(microbatch_offset)
        items = [
            simulate_from_key(
                int(pair_id),
                spec=self.spec,
                grid=self.grid,
                master_seed=self.master_seed,
                split=self.split,
                epoch=None,
            )
            for pair_id in pair_ids.tolist()
        ]
        batch = default_collate(items)
        # ``sample_id`` remains for compatibility with fixed-set artifacts;
        # this explicit name makes the training-stream identity unambiguous.
        batch["global_pair_id"] = batch["sample_id"]
        if self.include_metadata:
            return batch
        return {
            "global_pair_id": batch["global_pair_id"],
            "sample_id": batch["sample_id"],
            "histogram": batch["histogram"],
            "measurement": batch["measurement"],
        }

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            microbatch_offset = 0
            worker_count = 1
        else:
            microbatch_offset = worker.id
            worker_count = worker.num_workers
        while True:
            yield self._simulate_microbatch(microbatch_offset)
            microbatch_offset += worker_count


def generate_fixed_split(
    *,
    num_samples: int,
    spec: ContinuousNMRSpec,
    master_seed: int,
    split: str,
) -> dict[str, Any]:
    dataset = ContinuousGammaMixtureDataset(
        num_samples=num_samples,
        spec=spec,
        master_seed=master_seed,
        split=split,
        refresh_each_epoch=False,
    )
    items = [dataset[index] for index in range(num_samples)]
    keys = tuple(items[0])
    artifact: dict[str, Any] = {
        key: torch.stack([item[key] for item in items], dim=0) for key in keys
    }
    artifact.update(
        {
            "protocol_id": spec.protocol_id,
            "specification": spec.to_dict(),
            "specification_hash": spec.specification_hash,
            "split": split,
            "master_seed": int(master_seed),
            "grid": dataset.grid.as_artifact(),
        }
    )
    return artifact
