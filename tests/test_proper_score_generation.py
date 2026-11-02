"""Independent production truth splits, exact partitions, and immutable caches."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, SequentialSampler

CODES_DIR = Path(__file__).resolve().parents[1] / "codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from data.simulator import ContinuousGammaMixtureDataset  # noqa: E402
from data.spec import ContinuousNMRSpec  # noqa: E402
from evaluate import generate_proper_score_samples as generation  # noqa: E402
from evaluate import prepare_proper_score_conditions as preparation  # noqa: E402
from utils.model_variant_utils import VARIANTS, resolve_training_dir  # noqa: E402


class _SingleProcess:
    device = torch.device("cpu")
    is_main_process = True
    num_processes = 1
    process_index = 0

    def prepare(self, value):
        return value

    def gather_for_metrics(self, value):
        return value

    def gather(self, value):
        return value

    def reduce(self, value, reduction):
        return value

    def free_memory(self):
        pass

    def print(self, *values, **kwargs):
        pass

    def autocast(self):
        return nullcontext()


class _FakeSampler:
    def __init__(self, info, accelerator, source_seed, max_decode_batch_size):
        self.info = info
        self.generator = torch.Generator().manual_seed(source_seed)

    def sample(self, measurement, num_samples, nfe):
        count = 1 if self.info["deterministic"] else num_samples
        values = torch.rand(len(measurement), count, self.info["spec"].num_spectrum_bins,
                            generator=self.generator)
        return values / values.sum(-1, keepdim=True)


def _args(root, variant="histogram_linear"):
    return generation.parse_args([
        "--experiment_dir", str(root / "m64n128"), "--variant", variant,
        "--run_dir", str(root / "run"), "--num_conditions", "16",
        "--num_posterior_samples", "8", "--eval_batch_size", "4",
        "--max_generated_batch_size", "32", "--mixed_precision", "no",
        "--eval_ode_steps", "1",
    ])


def _info(root, deterministic=False):
    spec = ContinuousNMRSpec(num_measurements=32, num_spectrum_bins=8)
    return {
        "spec": spec, "method": "deterministic_regression" if deterministic else "histogram_linear",
        "deterministic": deterministic, "checkpoint_identity": {"sha256": "fixed-weights"},
        "checkpoint": root / "checkpoint", "training_config": {}, "state_size": 8,
    }


def _prepare_truth(args, accelerator, spec):
    prepared = preparation.parse_args([
        "--experiment_dir", str(args.experiment_dir), "--run_dir", str(args.run_dir),
        "--num_conditions", str(args.num_conditions),
        "--num_posterior_samples", str(args.num_posterior_samples),
        "--num_measurements", str(spec.num_measurements),
        "--num_spectrum_bins", str(spec.num_spectrum_bins),
        "--test_dataset_seed", str(args.test_dataset_seed),
    ])
    preparation.prepare_conditions(prepared)
    return generation._load_truth(args, accelerator, spec)


class ProperScoreGenerationTests(unittest.TestCase):
    def test_direct_script_imports_without_project_on_pythonpath(self):
        # Importing in the test runner hides missing entrypoint path setup.
        # Isolated Python also ignores cwd/PYTHONPATH, as direct launch must
        # locate both the codes package and the existing top-level utilities.
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", str(CODES_DIR / "evaluate" / "generate_proper_score_samples.py"),
                 "--help"],
                cwd=directory, capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--num_conditions", result.stdout)
        self.assertIn("--num_posterior_samples", result.stdout)

    def test_production_defaults_and_unsupported_sampling_options(self):
        for variant, (method, *_) in VARIANTS.items():
            args = generation.parse_args(["--experiment_dir", "/tmp/m64n128", "--variant", variant])
            self.assertEqual((args.num_conditions, args.num_posterior_samples), (16384, 8))
            self.assertEqual(args.run_dir, Path("/tmp/m64n128/proper_scores/g16384_k8"))
            self.assertEqual(args.eval_ode_steps, (256,) if method in generation.FLOW_METHODS else (1,))
        with self.assertRaises(SystemExit):
            generation.parse_args(["--experiment_dir", "/tmp/m64n128", "--variant", "conditional_vae_z16",
                                   "--eval_ode_steps", "2"])
        with self.assertRaises(SystemExit):
            generation.parse_args(["--experiment_dir", "/tmp/m64n128", "--variant", "histogram_linear",
                                   "--num_posterior_samples", "1"])

    def test_exact_four_rank_partitions_and_generated_batch_limit(self):
        for count, requested, draws, cap in ((16384, 64, 8, 512), (28, 5, 3, 32), (4, 64, 8, 512)):
            size = generation.effective_batch_size(count, 4, requested, draws, cap)
            self.assertLessEqual(size * draws, cap)
            sampler = BatchSampler(SequentialSampler(range(count)), size, drop_last=False)
            all_ids = []
            for rank in range(4):
                shard = BatchSamplerShard(sampler, num_processes=4, process_index=rank,
                                          split_batches=False, even_batches=True)
                ids = [index for batch in shard for index in batch]
                self.assertEqual(len(ids), count // 4)
                self.assertEqual(len(set(ids)), len(ids))
                all_ids.extend(ids)
            self.assertEqual(sorted(all_ids), list(range(count)))
        with self.assertRaisesRegex(ValueError, "divisible"):
            generation.effective_batch_size(15, 4, 64, 8, 512)
        with self.assertRaisesRegex(ValueError, "fit at least one"):
            generation.effective_batch_size(16, 4, 64, 513, 512)

    def test_fresh_truth_namespace_and_byte_identical_reuse_across_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, info, accelerator = _args(root), _info(root), _SingleProcess()
            truth = _prepare_truth(args, accelerator, info["spec"])
            path = args.run_dir / "test_conditions.pt"
            initial_bytes = generation.file_sha256(path)
            self.assertEqual(truth["split"], "proper_score_test")
            self.assertNotIn("sweep_provenance", truth)
            for split in ("train", "score_sweep_test", "kernel_score_calibration"):
                other = ContinuousGammaMixtureDataset(num_samples=16, spec=info["spec"],
                    master_seed=args.test_dataset_seed, split=split, refresh_each_epoch=False)[0]
                self.assertFalse(torch.equal(truth["measurement"][0], other["measurement"]))
            other_args = _args(root, "conditional_vae_z16")
            other_args.reuse_existing = False  # Only prediction reuse is optional.
            with patch.object(preparation, "ContinuousGammaMixtureDataset", side_effect=AssertionError("must reuse truths")):
                reused = generation._load_truth(other_args, accelerator, info["spec"])
            self.assertEqual(generation.file_sha256(path), initial_bytes)
            torch.testing.assert_close(reused["histogram"], truth["histogram"], rtol=0, atol=0)

    def test_missing_truth_is_rejected_without_implicit_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, info, accelerator = _args(root), _info(root), _SingleProcess()
            with patch.object(preparation, "ContinuousGammaMixtureDataset", side_effect=AssertionError("no implicit simulation")):
                with self.assertRaisesRegex(RuntimeError, "00_prepare_test_conditions"):
                    generation._load_truth(args, accelerator, info["spec"])
            self.assertFalse((args.run_dir / "test_conditions.pt").exists())

    def test_prediction_cache_reuse_and_checkpoint_seed_setting_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, info, accelerator = _args(root), _info(root), _SingleProcess()
            truth = _prepare_truth(args, accelerator, info["spec"])
            legacy_path = args.run_dir / "generation_manifest.json"
            legacy_bytes = json.dumps({"protocol_version": generation.PROTOCOL, "jobs": {"previous": {}}}).encode()
            legacy_path.write_bytes(legacy_bytes)
            with patch.object(generation, "PredictiveModel", _FakeSampler):
                generation._generate_job(args, accelerator, "model", info, 1, truth)
            self.assertEqual(legacy_path.read_bytes(), legacy_bytes)
            job_manifest = json.loads((args.run_dir / "model/nfe_0001/generation_manifest.json").read_text())
            self.assertEqual(list(job_manifest["jobs"]), ["model/nfe_0001"])
            path = args.run_dir / "model/nfe_0001/samples.pt"
            artifact = generation._verified_cache(path, True)
            self.assertEqual(artifact["samples"].shape, (16, 8, 8))
            self.assertFalse(artifact["posterior_reference_samples_used"])
            self.assertNotIn("sweep_provenance", artifact)
            self.assertEqual(artifact["rank_condition_counts"], [16])
            with patch.object(generation, "PredictiveModel", side_effect=AssertionError("cache must bypass forward")):
                generation._generate_job(args, accelerator, "model", info, 1, truth)
            for field in ("source_seed", "max_decode_batch_size"):
                old = getattr(args, field)
                setattr(args, field, old + 1)
                with self.assertRaisesRegex(RuntimeError, "different checkpoint, seed"):
                    generation._generate_job(args, accelerator, "model", info, 1, truth)
                setattr(args, field, old)
            changed = {**info, "checkpoint_identity": {"sha256": "modified-weights"}}
            with self.assertRaisesRegex(RuntimeError, "different checkpoint, seed"):
                generation._generate_job(args, accelerator, "model", changed, 1, truth)
            args.reuse_existing = False
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                generation._generate_job(args, accelerator, "model", info, 1, truth)

    def test_regression_stores_one_draw_and_preserves_requested_k_for_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root, "deterministic_regression")
            info, accelerator = _info(root, True), _SingleProcess()
            truth = _prepare_truth(args, accelerator, info["spec"])
            with patch.object(generation, "PredictiveModel", _FakeSampler):
                generation._generate_job(args, accelerator, "regression", info, 1, truth)
            artifact = generation._verified_cache(args.run_dir / "regression/samples.pt", True)
            self.assertEqual(artifact["samples"].shape, (16, 1, 8))
            self.assertEqual(artifact["proper_score_provenance"]["stored_sample_count"], 1)
            self.assertEqual(artifact["proper_score_provenance"]["requested_sample_count"], 8)
            self.assertEqual(artifact["requested_num_samples_per_condition"], 8)
            self.assertTrue(artifact["deterministic"])

    def test_truth_cache_detects_changed_seed_and_corrupted_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, info, accelerator = _args(root), _info(root), _SingleProcess()
            _prepare_truth(args, accelerator, info["spec"])
            args.test_dataset_seed += 1
            with self.assertRaisesRegex(RuntimeError, "different proper-score settings"):
                generation._load_truth(args, accelerator, info["spec"])
            path = args.run_dir / "test_conditions.pt"
            with path.open("ab") as handle:
                handle.write(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                generation._verified_cache(path, True)

    def test_all_variant_names_resolve_and_custom_model_contract_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "m64n128"
            for variant, (method, pattern, field, value) in VARIANTS.items():
                training = root / pattern.format(shape="m64_n128")
                training.mkdir(parents=True)
                config = {"experiment": method}
                if field:
                    config[field] = value
                (training / "training_config.json").write_text(json.dumps(config))
                self.assertEqual(resolve_training_dir(root, variant), training)
            with self.assertRaisesRegex(ValueError, "direct child"):
                resolve_training_dir(root, "conditional_vae_z16", root.parent / "elsewhere")
            wrong = root / "conditional_vae_m64_n128_unet64x4_z32"
            with self.assertRaisesRegex(ValueError, "latent_dim=16"):
                resolve_training_dir(root, "conditional_vae_z16", wrong)


if __name__ == "__main__":
    unittest.main()
