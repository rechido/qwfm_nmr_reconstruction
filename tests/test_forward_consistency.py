"""Offline forward errors, noise cancellation, and saved-artifact integrity."""

from __future__ import annotations

import copy
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from codes.data import ContinuousNMRSpec, generate_fixed_split
from codes.evaluate import evaluate_forward_consistency as evaluation
from codes.metrics.forward_consistency import (
    per_sample_forward_errors, mean_spectrum_forward_errors, summarize_forward_errors, METRICS,
)
from codes.utils.predictive_generation_utils import (
    PROPER_SCORE_PROTOCOL_VERSION, PROPER_SCORE_TEST_SPLIT, _save_cache, file_sha256,
)


class ForwardErrorMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = torch.tensor([[1., .5], [.4, .2]], dtype=torch.float64)
        self.truth = torch.tensor([[.5, .5], [.25, .75]], dtype=torch.float64)
        self.noise = torch.tensor([[.02, -.01], [-.01, .02]], dtype=torch.float64)
        self.clean = self.truth @ self.kernel.T
        self.observed = self.clean + self.noise

    def test_true_spectra_have_zero_clean_error_but_keep_observation_noise(self) -> None:
        errors = per_sample_forward_errors(
            self.truth[:, None], self.truth, self.observed, self.kernel, noise_standard_deviation=.01,
        )
        self.assertTrue(torch.equal(errors["clean_mse"], torch.zeros((2, 1), dtype=torch.float64)))
        torch.testing.assert_close(errors["noisy_mse"][:, 0], self.noise.square().mean(-1))
        torch.testing.assert_close(errors["noisy_normalized_mse"][:, 0], torch.full((2,), 2.5, dtype=torch.float64))

    def test_same_noise_cancels_and_all_draws_are_scored_before_averaging(self) -> None:
        samples = torch.tensor([[[1., 0.], [0., 1.]], [[.2, .8], [.8, .2]]], dtype=torch.float64)
        errors = per_sample_forward_errors(
            samples, self.truth, self.observed, self.kernel, noise_standard_deviation=.01, condition_batch_size=1,
        )
        same_noise_prediction = samples @ self.kernel.T + self.noise[:, None]
        expected_clean = (same_noise_prediction - self.observed[:, None]).square().mean(-1)
        torch.testing.assert_close(errors["clean_mse"], expected_clean)
        # The first condition's ensemble mean is the truth, but individual
        # predictions are both wrong. Averaging spectra first must not hide it.
        self.assertGreater(float(errors["clean_mse"][0].mean()), 0)
        torch.testing.assert_close(samples[0].mean(0), self.truth[0])
        larger_batch = per_sample_forward_errors(
            samples, self.truth, self.observed, self.kernel, noise_standard_deviation=.01, condition_batch_size=8,
        )
        for name in METRICS:
            torch.testing.assert_close(errors[name], larger_batch[name])

    def test_standard_error_uses_conditions_not_generated_draws(self) -> None:
        values = torch.tensor([[1., 1., 1.], [9., 9., 9.]], dtype=torch.float64)
        report = summarize_forward_errors({name: values for name in METRICS})
        self.assertEqual(report["noisy_mse"]["mean"], 5)
        self.assertAlmostEqual(report["noisy_mse"]["condition_standard_error"], 4)

    def test_mean_spectrum_error_is_separate_from_mean_individual_error(self) -> None:
        samples = torch.tensor([[[1., 0.], [0., 1.]], [[.2, .8], [.8, .2]]], dtype=torch.float64)
        individual = per_sample_forward_errors(samples, self.truth, self.observed, self.kernel,
                                               noise_standard_deviation=.01)
        ensemble = mean_spectrum_forward_errors(samples, self.truth, self.observed, self.kernel,
                                                noise_standard_deviation=.01)
        expected = ((samples.mean(1)-self.truth) @ self.kernel.T).square().mean(-1)/.01**2
        torch.testing.assert_close(ensemble["mean_spectrum_clean_normalized_mse"][:, 0], expected)
        self.assertEqual(float(expected[0]), 0)
        self.assertGreater(float(individual["clean_normalized_mse"][0].mean()), 0)
        self.assertTrue((expected <= individual["clean_normalized_mse"].mean(1)+1e-12).all())
        for count in (1,):
            one = mean_spectrum_forward_errors(samples[:, :count], self.truth, self.observed, self.kernel,
                                              noise_standard_deviation=.01)
            direct = per_sample_forward_errors(samples[:, :count], self.truth, self.observed, self.kernel,
                                               noise_standard_deviation=.01)
            for name in METRICS:
                torch.testing.assert_close(one["mean_spectrum_"+name], direct[name])

    def test_invalid_spectra_are_not_silently_normalized_or_clipped(self) -> None:
        samples = self.truth[:, None].clone()
        for bad in (samples * 2, torch.full_like(samples, float("nan")),
                    torch.tensor([[[1.1, -.1]], [[.5, .5]]], dtype=torch.float64)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                per_sample_forward_errors(bad, self.truth, self.observed, self.kernel, noise_standard_deviation=.01)


class ForwardEvaluationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.experiment = Path(temporary.name) / "m64n8"
        self.run = self.experiment / "proper_scores" / "g4_k3"
        self.output = self.experiment / "forward_consistency" / "g4_k3"
        self.spec = ContinuousNMRSpec(num_measurements=64, num_spectrum_bins=8)
        self.truth = generate_fixed_split(num_samples=4, spec=self.spec, master_seed=92, split=PROPER_SCORE_TEST_SPLIT)
        self.truth["proper_score_provenance"] = {
            "protocol_version": PROPER_SCORE_PROTOCOL_VERSION, "num_conditions": 4,
            "specification_hash": self.spec.specification_hash, "master_seed": 92,
            "split": PROPER_SCORE_TEST_SPLIT,
        }
        self.truth_path = self.run / "test_conditions.pt"
        _save_cache(self.truth_path, self.truth)
        self.sources = {}
        for variant in ("quantile_wasserstein_l128", "deterministic_regression_mse_w1"):
            self._create_model(variant)

    def _create_model(self, variant: str, nfe: int = 256) -> None:
        method, pattern, _, _ = evaluation.VARIANTS[variant]
        deterministic = method.startswith("deterministic_regression")
        draws = 1 if deterministic else 3
        samples = self.truth["histogram"][:, None, :].repeat(1, draws, 1)
        if not deterministic:
            samples[:, 1] = samples[:, 1].roll(1, dims=-1)
            samples[:, 2] = samples[:, 2].roll(-1, dims=-1)
        relative = Path(pattern.format(shape="m64_n8"))
        nfe = None if method not in evaluation.FLOW_METHODS else nfe
        if nfe is not None:
            relative /= f"nfe_{nfe:04d}"
        path = self.run / relative / "samples.pt"
        artifact = {
            "protocol_id": self.spec.protocol_id, "specification": self.spec.to_dict(),
            "specification_hash": self.spec.specification_hash, "grid": self.truth["grid"],
            "condition_ids": self.truth["sample_id"].clone(), "measurements": self.truth["measurement"].clone(),
            "samples": samples, "method": method, "deterministic": deterministic,
            "sampling_config": {"ode_steps": nfe} if nfe is not None else {},
            "proper_score_provenance": {
                "protocol_version": PROPER_SCORE_PROTOCOL_VERSION, "variant": variant, "method": method,
                "nfe": nfe, "conditions_sha256": file_sha256(self.truth_path),
                "requested_sample_count": 3, "stored_sample_count": draws,
                "checkpoint_identity": {"sha256": "fixture"},
            },
        }
        _save_cache(path, artifact)
        self.sources[variant] = (path, artifact)

    def _args(self) -> list[str]:
        return ["--run_dir", str(self.run), "--output_dir", str(self.output),
                "--variants", ",".join(self.sources), "--eval_ode_steps", "256",
                "--num_conditions", "4", "--sample_count", "3", "--num_cpu_threads", "1",
                "--condition_batch_size", "2"]

    def _evaluate(self, *extra: str) -> dict:
        # The evaluation must neither call any network nor generate data.
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("Unexpected neural inference")), \
                patch("codes.data.simulator.simulate_from_key", side_effect=AssertionError("Unexpected test simulation")):
            return evaluation.main([*self._args(), *extra])

    def _single_model_args(self, variant: str) -> list[str]:
        return ["--experiment_dir", str(self.experiment), "--variant", variant,
                "--run_dir", str(self.run), "--output_dir", str(self.output),
                "--num_conditions", "4", "--sample_count", "3", "--num_cpu_threads", "1"]

    def test_complete_workflow_outputs_both_metrics_and_keeps_all_inputs(self) -> None:
        inputs = [self.truth_path, *(path for path, _ in self.sources.values())]
        before = {path: file_sha256(path) for path in inputs}
        summary = self._evaluate()
        self.assertEqual(summary["model_setting_count"], 2)
        self.assertFalse(summary["prediction_noise_added"])
        self.assertFalse(summary["network_forward_performed"])
        self.assertFalse(summary["test_data_generated"])
        self.assertFalse(summary["es_ks_computed"])
        self.assertFalse(summary["k_sweep_performed"])
        self.assertEqual(summary["ground_truth"]["aggregate"]["clean_mse"]["mean"], 0)
        for report in summary["records"]:
            path, source = self.sources[report["variant"]]
            draws = source["samples"].shape[1]
            self.assertEqual(report["sample_count"], draws)
            prediction = source["samples"].double() @ self.truth["grid"]["kernel"].T
            expected = (prediction - self.truth["measurement"][:, None].double()).square().mean()
            self.assertAlmostEqual(report["aggregate"]["noisy_mse"]["mean"], float(expected), places=14)
            model_output = Path(report["report_path"]).parent
            stored = torch.load(model_output / "per_sample_errors.pt", weights_only=False)
            self.assertEqual(stored["errors"]["clean_mse"].shape, (4, draws))
            self.assertEqual(stored["errors"]["mean_spectrum_clean_mse"].shape, (4, 1))
            expected_mean = ((source["samples"].double().mean(1)-self.truth["histogram"].double())
                             @ self.truth["grid"]["kernel"].double().T).square().mean()/.005**2
            self.assertAlmostEqual(report["aggregate"]["mean_spectrum_clean_normalized_mse"]["mean"],
                                   float(expected_mean), places=12)
            with (model_output / "per_condition.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
        with (self.output / "forward_consistency_summary.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)
        self.assertEqual(json.loads((self.output / "forward_consistency_summary.json").read_text()), summary)
        for path, digest in before.items():
            self.assertEqual(file_sha256(path), digest)

    def test_reuses_matching_reports_without_recomputing_model_errors(self) -> None:
        first = self._evaluate()
        real = evaluation.per_sample_forward_errors
        with patch.object(evaluation, "per_sample_forward_errors", wraps=real) as compute:
            second = self._evaluate()
        self.assertEqual(compute.call_count, 1)  # Ground-truth baseline only.
        self.assertEqual(first, second)

    def test_model_specific_calls_preserve_previously_evaluated_models_and_nfe(self) -> None:
        variant = "quantile_wasserstein_l128"
        self._create_model(variant, nfe=4)
        args = self._single_model_args(variant)
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("Unexpected neural inference")), \
                patch("codes.data.simulator.simulate_from_key", side_effect=AssertionError("Unexpected test simulation")):
            first = evaluation.main([*args, "--nfe", "4"])
            self.assertEqual([report["nfe"] for report in first["records"]], [4])
            second = evaluation.main([*args, "--nfe", "256"])
            self.assertEqual([report["nfe"] for report in second["records"]], [4, 256])
            third = evaluation.main(self._single_model_args("deterministic_regression_mse_w1"))
            self.assertEqual(third["model_setting_count"], 3)
            # No filter means all saved NFE settings, not an assumed five-point sweep.
            fourth = evaluation.main(args)
            self.assertEqual(third, fourth)

    def test_different_models_can_score_concurrently_without_losing_comparison_rows(self) -> None:
        barrier = threading.Barrier(2)
        real_score = evaluation._score

        def synchronized_score(*args, **kwargs):
            barrier.wait(timeout=10)
            return real_score(*args, **kwargs)

        torch.set_num_threads(1)
        with patch.object(evaluation, "_score", side_effect=synchronized_score), \
                patch.object(torch, "set_num_threads"), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluation.main, self._single_model_args(variant)) for variant in self.sources]
            for future in futures:
                future.result(timeout=20)
        summary = json.loads((self.output / "forward_consistency_summary.json").read_text())
        self.assertEqual(summary["model_setting_count"], 2)
        self.assertEqual(set(summary["variants"]), set(self.sources))
        with (self.output / "forward_consistency_summary.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_comparison_does_not_silently_retain_stale_unselected_predictions(self) -> None:
        self._evaluate()
        variant = "quantile_wasserstein_l128"
        path, artifact = self.sources[variant]
        artifact["samples"] = artifact["samples"].roll(1, dims=-1)
        _save_cache(path, artifact)
        with self.assertRaisesRegex(ValueError, "Prediction source changed after scoring"):
            evaluation.main(self._single_model_args("deterministic_regression_mse_w1"))
        result = evaluation.main([*self._single_model_args(variant), "--recompute"])
        self.assertEqual(result["model_setting_count"], 2)

    def test_standard_cli_defaults_to_separate_metric_comparison_directory(self) -> None:
        args = evaluation.parse_args([
            "--experiment_dir", str(self.experiment), "--variant", "histogram_linear",
        ])
        self.assertEqual(args.variants, ("histogram_linear",))
        self.assertEqual(args.run_dir, self.experiment / "proper_scores" / "g16384_k8")
        self.assertEqual(args.output_dir, self.experiment / "metric_comparisons" / "forward_consistency" / "g16384_k8")
        self.assertIsNone(args.eval_ode_steps)

    def test_changed_sources_require_explicit_recompute(self) -> None:
        self._evaluate()
        path, artifact = self.sources["quantile_wasserstein_l128"]
        artifact["samples"] = artifact["samples"].roll(1, dims=-1)
        _save_cache(path, artifact)
        with self.assertRaisesRegex(ValueError, "Source/configuration changed"):
            self._evaluate()
        refreshed = self._evaluate("--recompute")
        self.assertEqual(refreshed["records"][0]["samples_sha256"], file_sha256(path))

    def test_input_replacement_during_loading_is_rejected(self) -> None:
        real_load = evaluation._verified_cache

        def replace_after_loading(path, reuse):
            artifact = real_load(path, reuse)
            replacement = copy.deepcopy(artifact)
            replacement["noise"] += .001
            replacement["measurement"] = replacement["clean_measurement"] + replacement["noise"]
            _save_cache(path, replacement)
            return artifact

        with patch.object(evaluation, "_verified_cache", side_effect=replace_after_loading):
            with self.assertRaisesRegex(ValueError, "Saved input changed while it was being read"):
                self._evaluate()

    def test_missing_output_checksum_is_not_silently_accepted(self) -> None:
        summary = self._evaluate()
        path = Path(summary["records"][0]["report_path"])
        report = json.loads(path.read_text())
        del report["output_sha256"]["per_sample_errors.pt"]
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Missing output checksums"):
            self._evaluate()

    def test_old_forward_summary_requires_explicit_recompute(self) -> None:
        self._evaluate()
        path = self.output / "forward_consistency_summary.json"
        old = json.loads(path.read_text())
        del old["mean_spectrum_protocol_version"]
        path.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError, "use --recompute"):
            self._evaluate()
        self.assertIn("mean_spectrum_protocol_version", self._evaluate("--recompute"))

    def test_changed_truth_does_not_partially_overwrite_existing_reports(self) -> None:
        self._evaluate()
        baseline = self.output / "ground_truth.json"
        summary_path = self.output / "forward_consistency_summary.json"
        before = {path: path.read_bytes() for path in (baseline, summary_path)}
        changed = copy.deepcopy(self.truth)
        changed["noise"] += .001
        changed["measurement"] = changed["clean_measurement"] + changed["noise"]
        _save_cache(self.truth_path, changed)
        with self.assertRaisesRegex(ValueError, "Test conditions/evaluation configuration changed"):
            self._evaluate()
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_rejects_misaligned_observations_conditions_and_physics(self) -> None:
        path, original = self.sources["quantile_wasserstein_l128"]
        changes = {
            "measurements": original["measurements"] + .01,
            "condition_ids": original["condition_ids"].flip(0),
            "grid": {**original["grid"], "kernel": original["grid"]["kernel"] * .99},
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                artifact = {**original, field: value}
                _save_cache(path, artifact)
                with self.assertRaises(ValueError):
                    self._evaluate()
        _save_cache(path, original)

    def test_rejects_corrupt_truth_noise_and_missing_predictions(self) -> None:
        bad = copy.deepcopy(self.truth)
        bad["noise"] += .01
        _save_cache(self.truth_path, bad)
        with self.assertRaisesRegex(ValueError, "clean=A h"):
            self._evaluate()
        _save_cache(self.truth_path, self.truth)
        path, _ = self.sources["quantile_wasserstein_l128"]
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "No inference"):
            self._evaluate()
        self.assertFalse(path.exists())

    def test_refuses_reports_inside_saved_input_directory(self) -> None:
        with self.assertRaises(SystemExit):
            evaluation.parse_args([*self._args(), "--output_dir", str(self.run / "reports")])


if __name__ == "__main__":
    unittest.main()
