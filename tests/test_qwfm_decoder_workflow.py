"""Paired decoder generation/re-decoding and real offline scoring contracts."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_proper_score_generation import _SingleProcess, _prepare_truth
from codes.evaluate import generate_proper_score_samples as generation
from codes.evaluate import decode_qwfm_samples as decoding
from codes.evaluate import evaluate_proper_scores as scores
from codes.data.spec import ContinuousNMRSpec
from codes.data.grid import build_nmr_grid
from codes.utils.nmr_utils import decode_quantiles, make_normalized_log_rate_edges
from codes.utils.predictive_generation_utils import _verified_cache, file_sha256
from codes.utils.quantile_decoder_artifacts import reuse_test_conditions, validate_saved_quantiles


class _QuantileFixture:
    """Seeded lightweight state fixture, deliberately no learned network."""
    calls = 0

    def __init__(self, info, accelerator, source_seed, max_decode_batch_size):
        self.info = info
        self.generator = torch.Generator().manual_seed(source_seed)
        self.edges = make_normalized_log_rate_edges(build_nmr_grid(info["spec"]).log_rate_edges)

    def sample(self, measurements, num_samples, nfe, return_quantiles=False):
        type(self).calls += 1
        q = torch.rand(len(measurements), num_samples, self.info["state_size"],
                       generator=self.generator).sort(-1).values
        h = decode_quantiles(q.flatten(0, 1), self.edges).reshape(
            len(measurements), num_samples, self.info["spec"].num_spectrum_bins)
        return (h, q) if return_quantiles else h


class DecoderWorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "m64n128"
        self.original = self.root / "proper_scores" / "g4_k8"
        self.hard = self.root / "decoder_comparison" / "hard_bin" / "proper_scores" / "g4_k8"
        self.linear = self.root / "decoder_comparison" / "linear_cdf" / "proper_scores" / "g4_k8"
        self.name = "quantile_wasserstein_m64_n128_l128_unet64x4_t1024"
        self.variant = "quantile_wasserstein_l128"
        self.args = generation.parse_args([
            "--experiment_dir", str(self.root), "--variant", self.variant,
            "--run_dir", str(self.original), "--num_conditions", "4",
            "--num_posterior_samples", "8", "--eval_batch_size", "2",
            "--max_generated_batch_size", "16", "--mixed_precision", "no",
            "--eval_ode_steps", "1", "--save_quantiles", "true",
        ])
        self.spec = ContinuousNMRSpec(num_measurements=64, num_spectrum_bins=128)
        self.accelerator = _SingleProcess()
        self.truth = _prepare_truth(self.args, self.accelerator, self.spec)
        self.original_hash = file_sha256(self.original / "test_conditions.pt")
        reuse_test_conditions(self.original, self.hard)
        self.args.run_dir = self.hard
        self.info = {"spec": self.spec, "method": "quantile_wasserstein",
                     "deterministic": False, "checkpoint_identity": {"sha256": "fixture-weights"},
                     "checkpoint": self.root / "checkpoint", "training_config": {"num_quantiles": 128},
                     "state_size": 128}
        with patch.object(generation, "PredictiveModel", _QuantileFixture):
            generation._generate_job(self.args, self.accelerator, self.name, self.info, 1, self.truth)
        self.relative = Path(self.name) / "nfe_0001" / "samples.pt"
        self.hard_path = self.hard / self.relative

    def decode(self):
        return decoding.main([
            "--source_run_dir", str(self.hard), "--output_run_dir", str(self.linear),
            "--variant", self.variant, "--num_conditions", "4", "--num_cpu_threads", "1",
        ])

    def test_saved_quantiles_are_paired_and_caches_preserved(self):
        before = file_sha256(self.hard_path)
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("No network")):
            paths = self.decode()
            self.assertEqual(paths, self.decode())
        hard = _verified_cache(self.hard_path, True)
        linear = _verified_cache(paths[0], True)
        torch.testing.assert_close(hard["quantiles"], linear["quantiles"], rtol=0, atol=0)
        self.assertEqual(hard["quantile_sha256"], linear["quantile_sha256"])
        self.assertFalse(torch.equal(hard["samples"], linear["samples"]))
        torch.testing.assert_close(linear["samples"].sum(-1), torch.ones(4, 8))
        self.assertEqual(before, file_sha256(self.hard_path))
        self.assertEqual(self.original_hash, file_sha256(self.original / "test_conditions.pt"))
        self.assertTrue((self.linear / "test_conditions.pt").is_symlink())
        self.assertEqual(hard["proper_score_provenance"]["derived_source_seed"],
                         linear["proper_score_provenance"]["derived_source_seed"])

    def test_saved_cache_reuses_without_another_forward(self):
        with patch.object(generation, "PredictiveModel", side_effect=AssertionError("No new inference")):
            generation._generate_job(self.args, self.accelerator, self.name, self.info, 1, self.truth)

    def test_other_variant_without_checksum_does_not_block_selected_decoder(self):
        other = self.hard / "quantile_wasserstein_m64_n128_l256_unet64x4_t1024" / "nfe_0001" / "samples.pt"
        other.parent.mkdir(parents=True)
        torch.save({"proper_score_provenance": {"variant": "quantile_wasserstein_l256"}}, other)
        self.assertFalse(other.with_suffix(".sha256.json").exists())
        paths = self.decode()
        self.assertEqual(paths, [self.linear / self.relative])
        self.assertFalse((self.linear / other.relative_to(self.hard)).exists())
        self.assertFalse(other.with_suffix(".sha256.json").exists())

    def test_custom_training_directory_name_remains_supported(self):
        custom = self.hard / "custom_qwfm_training"
        (self.hard / self.name).rename(custom)
        paths = self.decode()
        self.assertEqual(paths, [self.linear / "custom_qwfm_training" / "nfe_0001" / "samples.pt"])

    def test_selected_variant_still_requires_its_checksum(self):
        self.hard_path.with_suffix(".sha256.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.decode()
        self.assertFalse((self.linear / self.relative).exists())

    def test_missing_and_tampered_quantiles_rejected(self):
        artifact = _verified_cache(self.hard_path, True)
        missing = dict(artifact)
        missing.pop("quantiles")
        with self.assertRaises(ValueError):
            decoding.decode_artifact(missing)
        changed = copy.deepcopy(artifact)
        changed["quantiles"][0, 0, 0] *= .5
        with self.assertRaises(ValueError):
            validate_saved_quantiles(changed)

    def test_proper_scores_consume_both_decoders(self):
        self.decode()
        for decoder, run in (("hard_bin", self.hard), ("linear_cdf", self.linear)):
            output = self.root / "decoder_comparison" / decoder / "metric_comparisons"
            with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("No network")):
                summary = scores.main([
                    "--experiment_dir", str(self.root), "--run_dir", str(run),
                    "--output_dir", str(output / "proper_scores" / "g4_k8"), "--variant", self.variant,
                    "--num_conditions", "4", "--sample_count", "8", "--num_cpu_threads", "1",
                    "--kernel_bandwidth", ".2", "--kernel_bandwidth_w2", ".5", "--skip_plots",
                ])
            for metric in ("energy_score", "energy_score_w2", "kernel_score", "kernel_score_w2"):
                self.assertTrue((output / "proper_scores" / "g4_k8" / f"{metric}_summary.json").is_file())
            self.assertIsNotNone(summary)


if __name__ == "__main__":
    unittest.main()
