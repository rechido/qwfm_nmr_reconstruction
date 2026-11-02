"""Paired PR decoder comparisons reuse states and keep legacy results immutable."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from codes.evaluate import evaluate_quantile_wasserstein as evaluation
from codes.data import ContinuousNMRSpec
from tests.test_training_sampling_integration import _reference_pair


class _SingleProcess:
    device = torch.device("cpu")
    is_main_process = True
    process_index = 0

    def reduce(self, value, reduction):
        return value

    def gather(self, value):
        return value

    def gather_for_metrics(self, value):
        return value

    def wait_for_everyone(self):
        pass


class _Sampler:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def sample(self, condition, num_samples_per_condition, num_ode_steps, return_quantiles=False):
        assert return_quantiles
        type(self).calls += 1
        levels = (torch.arange(condition.shape[-1], dtype=torch.float32) + .5) / condition.shape[-1]
        q = levels.pow(1.3 + type(self).calls * .1).expand(
            len(condition), num_samples_per_condition, -1,
        ).clone()
        return torch.empty(len(condition), num_samples_per_condition, 8), {
            "quantiles": q,
            "pre_isotonic_projection_crossing_count": torch.tensor(0.),
            "pre_isotonic_projection_inspected_pair_count": torch.tensor(28.),
            "pre_clipping_out_of_domain_count": torch.tensor(0.),
            "pre_clipping_inspected_value_count": torch.tensor(32.),
        }


class QuantileDecoderPRTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = ContinuousNMRSpec(num_measurements=32, num_spectrum_bins=8)
        self.primary, self.oracle = _reference_pair(self.spec)
        self.dataset = evaluation.PosteriorConditionDataset(self.primary)
        self.args = argparse.Namespace(**self.spec.to_dict())
        self.args.__dict__.update(
            output_dir=str(self.root), checkpoint_path=str(self.root / "checkpoint"),
            model_subfolder="unet_ema", eval_output_subdir="decoder_comparison",
            primary_reference=str(self.root / "reference.pt"),
            oracle_reference=str(self.root / "oracle.pt"),
            num_quantiles=8, eval_batch_size=1, dataloader_num_workers=0,
            num_posterior_samples=4, num_visual_samples=2, eval_ode_steps=(4,),
            mixed_precision="no", seed=42, source_seed=43, model_time_scale=1024,
            reference_repeats=2, nearest_k=1, w2_num_quantiles=16,
            distance_chunk_size=8, metric_seed=99, credible_interval_level=.9,
            kernel_bandwidth=None, reuse_existing=False, compare_decoders=True,
            peak_relative_threshold=.05, _world_size=1,
        )
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        _Sampler.calls = 0

    def _path(self, decoder):
        return self.root / "decoder_comparison" / decoder / "nfe_0004" / "samples.pt"

    def _load(self, decoder):
        return torch.load(self._path(decoder), map_location="cpu", weights_only=False)

    def _run(self, allow_sampler=True, nfe=4):
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                evaluation, "QuantileWassersteinSampler",
                _Sampler if allow_sampler else unittest.mock.Mock(side_effect=AssertionError("no forward")),
            ))
            metrics = stack.enter_context(patch.object(evaluation, "evaluate_and_write_metrics"))
            stack.enter_context(patch.object(evaluation, "save_reconstruction_figure"))
            stack.enter_context(patch.object(evaluation, "save_reconstruction_grid_figure"))
            stack.enter_context(evaluation._comparison_writer_lock(self.args, _SingleProcess()))
            evaluation._evaluate_ode_steps(
                self.args, _SingleProcess(), object(), self.loader, self.dataset,
                self.root / "checkpoint", {"global_iteration": 200000}, {},
                self.primary, self.oracle, nfe,
            )
            return metrics.call_args_list

    def test_one_generation_supplies_both_decoders_without_touching_legacy(self):
        legacy = self.root / "evaluation" / "nfe_0004" / "samples.pt"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"existing naive QWFM result")
        calls = self._run()
        self.assertEqual(_Sampler.calls, len(self.loader))
        self.assertEqual(legacy.read_bytes(), b"existing naive QWFM result")
        hard, linear = self._load("hard_bin"), self._load("linear_cdf")
        self.assertTrue(torch.equal(hard["quantiles"], linear["quantiles"]))
        self.assertEqual(hard["quantile_sha256"], linear["quantile_sha256"])
        self.assertFalse(torch.equal(hard["samples"], linear["samples"]))
        torch.testing.assert_close(linear["samples"].sum(-1), torch.ones(2, 4))
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0].args[1], calls[1].args[1])
        self.assertIs(calls[0].args[2], calls[1].args[2])
        self.assertEqual(calls[0].kwargs["settings"], calls[1].kwargs["settings"])
        for decoder in ("hard_bin", "linear_cdf"):
            summary = json.loads(self._path(decoder).with_name("sampling_summary.json").read_text())
            self.assertEqual(summary["quantile_sha256"], hard["quantile_sha256"])
            self.assertEqual(summary["decoder_comparison"]["decoder_name"], decoder)

    def test_fresh_run_refuses_to_overwrite_either_comparison(self):
        self._run()
        original = self._path("hard_bin").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "will not be overwritten"):
            self._run(allow_sampler=False)
        self.assertEqual(self._path("hard_bin").read_bytes(), original)

    def test_reuse_true_generates_first_pair_when_no_cache_exists(self):
        self.args.reuse_existing = True
        self._run()
        self.assertEqual(_Sampler.calls, len(self.loader))
        self.assertTrue(self._path("hard_bin").is_file())
        self.assertTrue(self._path("linear_cdf").is_file())

    def test_partially_cached_nfe_sweep_reuses_old_and_generates_missing(self):
        self._run()
        original = self._path("hard_bin").read_bytes()
        self.args.reuse_existing = True
        self._run(allow_sampler=False)
        self._run(nfe=16)
        self.assertEqual(_Sampler.calls, 2 * len(self.loader))
        self.assertEqual(self._path("hard_bin").read_bytes(), original)
        for decoder in ("hard_bin", "linear_cdf"):
            self.assertTrue((self.root / "decoder_comparison" / decoder / "nfe_0016" / "samples.pt").is_file())

    def test_orphan_linear_cache_is_not_overwritten_or_regenerated(self):
        self._run()
        self._path("hard_bin").unlink()
        self.args.reuse_existing = True
        with self.assertRaisesRegex(RuntimeError, "will not be overwritten"):
            self._run(allow_sampler=False)

    def test_another_writer_is_rejected_before_sampling_and_lock_recovers_after_error(self):
        with evaluation._comparison_writer_lock(self.args, _SingleProcess()):
            with self.assertRaisesRegex(RuntimeError, "Another process is writing"):
                self._run(allow_sampler=False)
        with self.assertRaisesRegex(ValueError, "failed work"):
            with evaluation._comparison_writer_lock(self.args, _SingleProcess()):
                raise ValueError("failed work")
        self._run()
        self.assertEqual(_Sampler.calls, len(self.loader))

    def test_reuse_needs_no_model_forward_and_can_recover_missing_linear_artifact(self):
        self._run()
        original = self._path("hard_bin").read_bytes()
        self._path("linear_cdf").unlink()
        self.args.reuse_existing = True
        self._run(allow_sampler=False)
        self.assertEqual(self._path("hard_bin").read_bytes(), original)
        self.assertTrue(torch.equal(self._load("hard_bin")["quantiles"],
                                    self._load("linear_cdf")["quantiles"]))
        self._run(allow_sampler=False)

    def test_histogram_only_cache_is_rejected_instead_of_regenerated(self):
        self._run()
        artifact = self._load("hard_bin")
        del artifact["quantiles"]
        torch.save(artifact, self._path("hard_bin"))
        self.args.reuse_existing = True
        with self.assertRaisesRegex(RuntimeError, "saved final quantiles"):
            self._run(allow_sampler=False)

    def test_mismatched_decoder_provenance_is_rejected(self):
        self._run()
        artifact = self._load("linear_cdf")
        artifact["sampling_config"]["histogram_decoder"] = evaluation.qwfm_decoder_signature("hard_bin")
        torch.save(artifact, self._path("linear_cdf"))
        self.args.reuse_existing = True
        with self.assertRaisesRegex(RuntimeError, "stale provenance"):
            self._run(allow_sampler=False)

    def test_cache_with_different_raw_states_is_rejected(self):
        self._run()
        artifact = self._load("linear_cdf")
        artifact["quantiles"] = artifact["quantiles"].square()
        torch.save(artifact, self._path("linear_cdf"))
        self.args.reuse_existing = True
        with self.assertRaisesRegex(RuntimeError, "recorded identity"):
            self._run(allow_sampler=False)


if __name__ == "__main__":
    unittest.main()
