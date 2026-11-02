"""Shared exact partitioning, checked caches, and inference progress reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import torch
from accelerate import Accelerator

from utils.io_utils import atomic_torch_save, atomic_write_json, load_torch_artifact


PROPER_SCORE_PROTOCOL_VERSION = "proper_score_generation_v1"
PROPER_SCORE_TEST_SPLIT = "proper_score_test"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "pending"
    total = max(0, math.ceil(seconds))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


class _GenerationProgress:
    """Report global completed conditions after the existing distributed gather.

    ETA concerns the current loop only. Model loading and final file writing are
    separate phases, so their unknown duration is not presented as a known ETA.
    """

    def __init__(self, accelerator: Accelerator, label: str, total: int,
                 every: int, interval_seconds: float) -> None:
        self.accelerator = accelerator
        self.label = label
        self.total = total
        self.every = every
        self.interval_seconds = interval_seconds
        self.started = self.last_report = time.perf_counter()
        accelerator.print(f"{label}: 0/{total} (0.0%) | elapsed=00:00:00 | eta=pending", flush=True)

    def update(self, completed: int, step: int) -> None:
        now = time.perf_counter()
        if not (step == 1 or completed == self.total or step % self.every == 0
                or now - self.last_report >= self.interval_seconds):
            return
        elapsed = max(0.0, now - self.started)
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - completed) / rate if rate else None
        self.accelerator.print(
            f"{self.label}: {completed}/{self.total} ({100*completed/self.total:.1f}%)"
            f" | elapsed={_duration(elapsed)} | eta={_duration(remaining)}"
            f" | {rate:.2f} conditions/s", flush=True,
        )
        self.last_report = now


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _positive_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(v.strip()) for v in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.") from error
    if not result or min(result) < 1 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Values must be positive and unique.")
    return result


def effective_batch_size(
    count: int, world_size: int, requested: int, draws: int, maximum_states: int,
) -> int:
    """Use exact, equally sized rank partitions with no Accelerate padding."""
    if count < 1 or world_size < 1 or count % world_size:
        raise ValueError("num_conditions must be positive and divisible by num_processes.")
    if min(requested, draws, maximum_states) < 1 or draws > maximum_states:
        raise ValueError("The generated-batch limit must fit at least one condition.")
    rank_count = count // world_size
    limit = min(requested, rank_count, maximum_states // draws)
    return next(size for size in range(limit, 0, -1) if rank_count % size == 0)


def _collective(accelerator: Accelerator, action: Callable[[], Any], phase: str) -> Any:
    result, local_error = None, None
    try:
        result = action()
    except Exception as error:
        local_error = error
    failed = torch.tensor(int(local_error is not None), device=accelerator.device)
    count = int(accelerator.reduce(failed, reduction="sum").item())
    if count:
        raise RuntimeError(f"{phase} failed on {count} rank(s): {local_error or 'see another rank'}") from local_error
    return result


def _main_action(accelerator: Accelerator, action: Callable[[], Any], phase: str) -> None:
    _collective(accelerator, lambda: action() if accelerator.is_main_process else None, phase)


def _verified_cache(path: Path, reuse: bool) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not reuse:
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output directory or reuse_existing=true.")
    stamp = _read_json(path.with_suffix(".sha256.json"))
    if stamp.get("sha256") != file_sha256(path):
        raise ValueError(f"Cached artifact checksum differs: {path}")
    return load_torch_artifact(path)


def _save_cache(path: Path, artifact: dict[str, Any]) -> None:
    atomic_torch_save(path, artifact)
    atomic_write_json(path.with_suffix(".sha256.json"), {"sha256": file_sha256(path)})


def _derived_seed(master_seed: int, *tokens: Any) -> int:
    content = json.dumps([master_seed, *tokens], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "little") % (2**63 - 1)
