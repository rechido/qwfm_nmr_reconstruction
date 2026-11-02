"""Separate model writers may overlap without mutating their shared truths."""

from __future__ import annotations

from contextlib import redirect_stdout
import fcntl
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

CODES_DIR = Path(__file__).resolve().parents[1] / "codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from evaluate import prepare_proper_score_conditions as preparation  # noqa: E402
from utils.predictive_generation_utils import _verified_cache, file_sha256  # noqa: E402
from utils.proper_score_locks import file_lock, model_generation_lock  # noqa: E402


LOCK_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from utils.proper_score_locks import model_generation_lock
with model_generation_lock(Path(sys.argv[2]), sys.argv[3]):
    print("locked", flush=True)
    sys.stdin.readline()
"""

EXCEPTION_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from utils.proper_score_locks import model_generation_lock
try:
    with model_generation_lock(Path(sys.argv[2]), "model_a"):
        raise ValueError("deliberate error")
except ValueError:
    print("released", flush=True)
    sys.stdin.readline()
"""


def _command(run_dir: Path, name: str) -> list[str]:
    return [sys.executable, "-c", LOCK_SCRIPT, str(CODES_DIR), str(run_dir), name]


def _start(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _line(process: subprocess.Popen) -> str:
    readable, _, _ = select.select([process.stdout], [], [], 15)
    if not readable:
        raise AssertionError("Timed out waiting for lock subprocess.")
    return process.stdout.readline().strip()


def _finish(process: subprocess.Popen) -> tuple[str, str]:
    try:
        return process.communicate("\n", timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise


class ProperScoreConcurrencyTests(unittest.TestCase):
    def test_shared_lock_supports_record_lock_backend(self):
        # NFS implements flock through record locks, which require a readable
        # descriptor for LOCK_SH even though local flock accepts write-only FDs.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".generation.lock"
            for existing in (False, True):
                with self.subTest(existing=existing):
                    if existing:
                        path.write_text("preserve existing lock file\n")
                    with mock.patch("utils.proper_score_locks.fcntl.flock",
                                    side_effect=fcntl.lockf):
                        with file_lock(path, shared=True):
                            self.assertTrue(path.is_file())
                    expected = "preserve existing lock file\n" if existing else ""
                    self.assertEqual(path.read_text(), expected)

    def test_distinct_models_overlap_but_duplicate_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = []
            try:
                first = _start(_command(root, "model_a"))
                active.append(first)
                self.assertEqual(_line(first), "locked")
                second = _start(_command(root, "model_b"))
                active.append(second)
                self.assertEqual(_line(second), "locked")
                self.assertIsNone(first.poll())
                self.assertIsNone(second.poll())
                duplicate = subprocess.run(_command(root, "model_a"), input="\n", text=True,
                                           capture_output=True, timeout=15)
                self.assertNotEqual(duplicate.returncode, 0)
                self.assertIn("same model: model_a", duplicate.stderr)
            finally:
                for process in active:
                    _finish(process)
            for process in active:
                self.assertEqual(process.returncode, 0)
            # Both shared compatibility and exclusive model locks are released.
            with file_lock(root / ".generation.lock"):
                pass
            with model_generation_lock(root, "model_a"):
                pass

    def test_legacy_exclusive_generator_rejects_new_model_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with file_lock(root / ".generation.lock"):
                blocked = subprocess.run(_command(root, "model_b"), input="\n", text=True,
                                         capture_output=True, timeout=15)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("older generator", blocked.stderr)
            self.assertIn("do not delete the lock file", blocked.stderr)
            retry = subprocess.run(_command(root, "model_b"), input="\n", text=True,
                                   capture_output=True, timeout=15)
            self.assertEqual(retry.returncode, 0, retry.stderr)

    def test_both_locks_release_after_exception_while_process_stays_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = _start([sys.executable, "-c", EXCEPTION_SCRIPT, str(CODES_DIR), str(root)])
            try:
                self.assertEqual(_line(process), "released")
                self.assertIsNone(process.poll())
                # This must work before the child exits; OS cleanup at exit
                # alone would not establish exception-safe lock release.
                with file_lock(root / ".generation.lock"):
                    pass
                with model_generation_lock(root, "model_a"):
                    pass
            finally:
                _finish(process)
            self.assertEqual(process.returncode, 0)

    def test_two_classical_jobs_preserve_truth_and_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "m64n128"
            run_dir = experiment / "proper_scores" / "g2_k8"
            args = preparation.parse_args([
                "--experiment_dir", str(experiment), "--num_conditions", "2",
            ])
            with redirect_stdout(io.StringIO()):
                truth = preparation.prepare_conditions(args)
            truth_path = run_dir / "test_conditions.pt"
            original_hash = file_sha256(truth_path)
            original_mtime = truth_path.stat().st_mtime_ns
            root_manifest = run_dir / "generation_manifest.json"
            original_manifest = b'{"protocol_version":"proper_score_generation_v1","jobs":{"sentinel":{}}}\n'
            root_manifest.write_bytes(original_manifest)
            variants = ("classic_nnls", "classic_x2_d2")
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1"}
            processes = []
            try:
                for variant in variants:
                    command = [sys.executable, str(CODES_DIR / "evaluate" / "generate_proper_score_classical.py"),
                               "--experiment_dir", str(experiment), "--variant", variant,
                               "--num_conditions", "2", "--num_posterior_samples", "8",
                               "--num_workers", "1", "--condition_batch_size", "1", "--reuse_existing"]
                    processes.append(subprocess.Popen(command, stdout=subprocess.PIPE,
                                                      stderr=subprocess.STDOUT, text=True, env=environment))
                logs = [process.communicate(timeout=60)[0] for process in processes]
                for process, log in zip(processes, logs):
                    self.assertEqual(process.returncode, 0, log)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            self.assertEqual(root_manifest.read_bytes(), original_manifest)
            self.assertEqual(file_sha256(truth_path), original_hash)
            self.assertEqual(truth_path.stat().st_mtime_ns, original_mtime)
            for variant in variants:
                path = run_dir / variant / "samples.pt"
                artifact = _verified_cache(path, True)
                self.assertEqual(artifact["samples"].shape, (2, 1, 128))
                self.assertTrue(torch.equal(artifact["condition_ids"], truth["sample_id"]))
                self.assertTrue(torch.equal(artifact["measurements"], truth["measurement"]))
                self.assertEqual(artifact["proper_score_provenance"]["conditions_sha256"], original_hash)
                manifest = json.loads((path.parent / "generation_manifest.json").read_text())
                self.assertEqual(manifest["conditions_sha256"], original_hash)
                self.assertEqual(set(manifest["jobs"]), {variant})


if __name__ == "__main__":
    unittest.main()
