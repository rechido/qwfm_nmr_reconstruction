"""CPU truth preparation, immutable reuse, and read-only evaluation loading."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import fcntl
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

CODES_DIR = Path(__file__).resolve().parents[1] / "codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from data.simulator import ContinuousGammaMixtureDataset  # noqa: E402
from data.spec import ContinuousNMRSpec  # noqa: E402
from evaluate import prepare_proper_score_conditions as preparation  # noqa: E402
from utils.predictive_generation_utils import _save_cache, file_sha256  # noqa: E402
from utils.proper_score_conditions import load_proper_score_truth  # noqa: E402


def _args(root: Path, *extra: str):
    return preparation.parse_args([
        "--experiment_dir", str(root / "m64n128"), "--run_dir", str(root / "run"),
        "--num_conditions", "8", "--num_spectrum_bins", "8",
        "--condition_batch_size", "3", *extra,
    ])


def _prepare(args):
    with redirect_stdout(io.StringIO()):
        return preparation.prepare_conditions(args)


class ProperScoreConditionsTests(unittest.TestCase):
    def test_defaults_infer_experiment_without_model(self):
        for measurement_count, bins in ((32, 128), (64, 128), (64, 256), (128, 256)):
            args = preparation.parse_args(["--experiment_dir", f"/tmp/m{measurement_count}n{bins}"])
            self.assertEqual(args.spec, ContinuousNMRSpec(num_measurements=measurement_count,
                                                       num_spectrum_bins=bins))
            self.assertEqual((args.num_conditions, args.num_posterior_samples), (16384, 8))
            self.assertEqual(args.run_dir.name, "g16384_k8")
            self.assertEqual(args.num_cpu_threads, 2)
            self.assertFalse(hasattr(args, "variant"))
            self.assertFalse(hasattr(args, "checkpoint_path"))
        explicit = preparation.parse_args([
            "--experiment_dir", "/tmp/custom", "--num_measurements", "32", "--num_spectrum_bins", "16",
        ])
        self.assertEqual(explicit.spec.num_spectrum_bins, 16)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            preparation.parse_args(["--experiment_dir", "/tmp/custom"])

    def test_cpu_generation_matches_simulator_and_is_byte_identical_on_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            truth = _prepare(args)
            dataset = ContinuousGammaMixtureDataset(
                num_samples=8, spec=args.spec, master_seed=args.test_dataset_seed,
                split="proper_score_test", refresh_each_epoch=False,
            )
            for index in range(8):
                for field, value in dataset[index].items():
                    torch.testing.assert_close(truth[field][index], value, rtol=0, atol=0)
            self.assertEqual(truth["measurement"].device.type, "cpu")
            files = ("test_conditions.pt", "test_conditions.sha256.json", "preparation_manifest.json")
            original = {name: ((args.run_dir / name).read_bytes(), (args.run_dir / name).stat().st_mtime_ns)
                        for name in files}
            with patch.object(preparation, "ContinuousGammaMixtureDataset",
                              side_effect=AssertionError("Existing truth must not be recreated")):
                reused = _prepare(args)
            for name, (content, mtime) in original.items():
                self.assertEqual((args.run_dir / name).read_bytes(), content)
                self.assertEqual((args.run_dir / name).stat().st_mtime_ns, mtime)
            torch.testing.assert_close(reused["histogram"], truth["histogram"], rtol=0, atol=0)
            self.assertFalse((args.run_dir / "generation_manifest.json").exists())

    def test_loader_never_creates_missing_directory_and_gives_preparation_command(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            with self.assertRaisesRegex(FileNotFoundError, "00_prepare_test_conditions.sh"):
                load_proper_score_truth(args.run_dir, num_conditions=8,
                                        test_dataset_seed=args.test_dataset_seed)
            self.assertFalse(args.run_dir.exists())

    def test_truth_reuse_remains_read_only_while_other_model_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            _prepare(args)
            path = args.run_dir / "test_conditions.pt"
            original = file_sha256(path)
            with (args.run_dir / ".generation.lock").open("a") as active_model:
                fcntl.flock(active_model.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                _prepare(args)
                truth = load_proper_score_truth(args.run_dir, num_conditions=8,
                                               test_dataset_seed=args.test_dataset_seed)
            self.assertEqual(file_sha256(path), original)
            self.assertEqual(len(truth["sample_id"]), 8)

    def test_missing_truth_creation_rejects_active_legacy_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            args.run_dir.mkdir()
            with (args.run_dir / ".generation.lock").open("a") as old_generator:
                fcntl.flock(old_generator.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "legacy run lock"):
                    _prepare(args)
            self.assertFalse((args.run_dir / "test_conditions.pt").exists())

    def test_changed_settings_are_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            _prepare(args)
            initial = file_sha256(args.run_dir / "test_conditions.pt")
            for options in (("--num_conditions", "4"), ("--test_dataset_seed", "1"),
                            ("--num_measurements", "32"), ("--num_spectrum_bins", "16")):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "settings"):
                    _prepare(_args(root, *options))
            self.assertEqual(file_sha256(args.run_dir / "test_conditions.pt"), initial)

    def test_rejects_corruption_missing_checksum_and_orphan_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            _prepare(args)
            path = args.run_dir / "test_conditions.pt"
            checksum = path.with_suffix(".sha256.json")
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                _prepare(args)
            checksum_contents = checksum.read_bytes()
            checksum.unlink()
            with self.assertRaises(FileNotFoundError):
                _prepare(args)
            path.unlink()
            checksum.write_bytes(checksum_contents)
            with self.assertRaisesRegex(FileExistsError, "without its artifact"):
                _prepare(args)

    def test_rejects_internally_inconsistent_truth_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            truth = _prepare(args)
            path = args.run_dir / "test_conditions.pt"
            wrong_ids = {**truth, "sample_id": torch.arange(8).flip(0)}
            _save_cache(path, wrong_ids)
            with self.assertRaisesRegex(ValueError, "IDs"):
                load_proper_score_truth(args.run_dir, num_conditions=8,
                                        test_dataset_seed=args.test_dataset_seed)
            wrong_spec = {**truth, "specification": {**truth["specification"], "num_measurements": 32}}
            _save_cache(path, wrong_spec)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_proper_score_truth(args.run_dir, num_conditions=8,
                                        test_dataset_seed=args.test_dataset_seed, spec=args.spec)

    def test_two_concurrent_preparers_create_one_shared_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            command = [sys.executable, str(CODES_DIR / "evaluate" / "prepare_proper_score_conditions.py"),
                       "--experiment_dir", str(root / "m64n128"), "--run_dir", str(args.run_dir),
                       "--num_conditions", "8", "--num_spectrum_bins", "8"]
            processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True) for _ in range(2)]
            try:
                logs = [process.communicate(timeout=60)[0] for process in processes]
                for process, log in zip(processes, logs):
                    self.assertEqual(process.returncode, 0, log)
                self.assertEqual(sum("test conditions: saved" in log for log in logs), 1)
                self.assertEqual(sum("test conditions: reused" in log for log in logs), 1)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            load_proper_score_truth(args.run_dir, num_conditions=8,
                                    test_dataset_seed=args.test_dataset_seed, spec=args.spec)


if __name__ == "__main__":
    unittest.main()
