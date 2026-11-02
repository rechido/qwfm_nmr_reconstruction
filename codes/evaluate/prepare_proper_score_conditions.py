# ruff: noqa: E402
"""Prepare shared independent (y, h) test pairs before any ES/KS model evaluation.

This CPU-only step needs no trained model or checkpoint. Later generators read
the immutable test_conditions.pt file and write their own prediction files.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from data.simulator import ContinuousGammaMixtureDataset
from data.spec import ContinuousNMRSpec
from utils.io_utils import atomic_write_json
from utils.predictive_generation_utils import (
    PROPER_SCORE_PROTOCOL_VERSION,
    PROPER_SCORE_TEST_SPLIT,
    _duration,
    _GenerationProgress,
    _save_cache,
    file_sha256,
)
from utils.proper_score_conditions import load_proper_score_truth, validate_proper_score_truth
from utils.model_variant_utils import experiment_dimensions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--num_conditions", type=int, default=16384)
    parser.add_argument("--num_posterior_samples", type=int, default=8,
                        help="Used only to name the shared run directory; no model samples are generated.")
    parser.add_argument("--num_measurements", type=int, choices=(32, 64, 128),
                        help="Inferred from experiment_dir name m<M>n<N> when omitted.")
    parser.add_argument("--num_spectrum_bins", type=int,
                        help="Inferred from experiment_dir name m<M>n<N>; otherwise defaults to 128.")
    parser.add_argument("--test_dataset_seed", type=int, default=2026092901)
    parser.add_argument("--condition_batch_size", type=int, default=256)
    parser.add_argument("--num_cpu_threads", type=int, default=2)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--progress_interval_seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    for name in ("num_conditions", "num_posterior_samples", "condition_batch_size",
                 "num_cpu_threads", "progress_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive.")
    if not 0 <= args.test_dataset_seed < 2**63:
        parser.error("test_dataset_seed must lie in [0, 2**63).")
    if (not math.isfinite(args.progress_interval_seconds)
            or args.progress_interval_seconds <= 0):
        parser.error("progress_interval_seconds must be finite and positive.")
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    try:
        dimensions = experiment_dimensions(args.experiment_dir)
    except ValueError:
        dimensions = None
    if args.num_measurements is None:
        if dimensions is None:
            parser.error("Use experiment_dir named m<M>n<N>, or supply --num_measurements.")
        args.num_measurements = dimensions[0]
    if args.num_spectrum_bins is None:
        args.num_spectrum_bins = dimensions[1] if dimensions is not None else 128
    try:
        args.spec = ContinuousNMRSpec(num_measurements=args.num_measurements,
                                      num_spectrum_bins=args.num_spectrum_bins)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    args.run_dir = (args.run_dir or args.experiment_dir / "proper_scores" /
                    f"g{args.num_conditions}_k{args.num_posterior_samples}").expanduser().resolve()
    return args


def prepare_conditions(args: argparse.Namespace) -> dict[str, Any]:
    """Create truths once, or return the existing checked artifact unchanged."""
    torch.set_num_threads(args.num_cpu_threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "test_conditions.pt"
    started = time.perf_counter()
    # Only competing preparation commands take this lock. Model inference never
    # creates or modifies truths and therefore needs no experiment-wide lock.
    with (args.run_dir / ".test_conditions.lock").open("a") as lock, ExitStack() as guards:
        print("test conditions: waiting for preparation lock...", flush=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            truth = load_proper_score_truth(
                args.run_dir, num_conditions=args.num_conditions,
                test_dataset_seed=args.test_dataset_seed, spec=args.spec,
            )
            print(f"test conditions: reused {args.num_conditions} independent truths"
                  f" | elapsed={_duration(time.perf_counter() - started)}", flush=True)
            return truth
        if path.with_suffix(".sha256.json").exists():
            raise FileExistsError(
                f"Truth checksum exists without its artifact: {path}. "
                "Refusing to replace an incomplete cache; inspect it or choose a new run directory."
            )
        # An old running generator can still create this file while holding its
        # former whole-run lock. Do not race that process during the migration.
        legacy_lock = guards.enter_context((args.run_dir / ".generation.lock").open("a"))
        try:
            fcntl.flock(legacy_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Cannot create test conditions while a generator holds the legacy run lock. "
                "Let the running job finish, then run 00_prepare_test_conditions.sh again. "
                "Do not delete lock files."
            ) from error
        spec = args.spec
        dataset = ContinuousGammaMixtureDataset(
            num_samples=args.num_conditions, spec=spec, master_seed=args.test_dataset_seed,
            split=PROPER_SCORE_TEST_SPLIT, refresh_each_epoch=False,
        )
        loader = DataLoader(dataset, batch_size=args.condition_batch_size,
                            shuffle=False, num_workers=0)
        batches: dict[str, list[torch.Tensor]] = {}
        completed = 0
        progress = _GenerationProgress(SimpleNamespace(print=print), "test conditions",
                                       args.num_conditions, args.progress_every,
                                       args.progress_interval_seconds)
        for step, batch in enumerate(loader, start=1):
            for name, value in batch.items():
                batches.setdefault(name, []).append(value.detach().cpu())
            completed += len(batch["sample_id"])
            progress.update(completed, step)
        truth = {name: torch.cat(values) for name, values in batches.items()}
        expected = {
            "protocol_version": PROPER_SCORE_PROTOCOL_VERSION,
            "num_conditions": args.num_conditions,
            "specification_hash": spec.specification_hash,
            "master_seed": args.test_dataset_seed,
            "split": PROPER_SCORE_TEST_SPLIT,
        }
        truth.update(protocol_id=spec.protocol_id, specification=spec.to_dict(),
                     specification_hash=spec.specification_hash, split=PROPER_SCORE_TEST_SPLIT,
                     master_seed=args.test_dataset_seed, grid=dataset.grid.as_artifact(),
                     proper_score_provenance=expected)
        validate_proper_score_truth(truth, expected)
        print("test conditions: saving and checksumming...", flush=True)
        _save_cache(path, truth)
        atomic_write_json(args.run_dir / "preparation_manifest.json", {
            **expected, "conditions_sha256": file_sha256(path),
            "truth_path": str(path), "device": "cpu",
        })
        print(f"test conditions: saved {path}"
              f" | elapsed={_duration(time.perf_counter() - started)}", flush=True)
        return truth


def main(argv: list[str] | None = None) -> None:
    prepare_conditions(parse_args(argv))


if __name__ == "__main__":
    main()
