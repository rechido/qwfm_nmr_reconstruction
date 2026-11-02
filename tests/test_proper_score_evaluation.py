"""Independent ES/KS evaluation, cumulative comparisons, and cache separation."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from codes.data import ContinuousNMRSpec, generate_fixed_split
from codes.evaluate import evaluate_energy_score as legacy
from codes.evaluate import evaluate_proper_scores as evaluation
from codes.evaluate.plot_energy_score import validate_comparison_reports
from codes.metrics.distances import W2_EXACT_PROTOCOL_VERSION, pairwise_w2_exact
from codes.metrics import kernel_score as kernel_metrics
from codes.utils.io_utils import atomic_torch_save
from codes.utils.predictive_generation_utils import (
    PROPER_SCORE_PROTOCOL_VERSION, PROPER_SCORE_TEST_SPLIT, _derived_seed,
    _save_cache, file_sha256,
)


class ProperScoreEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "m64n128"
        self.run = self.root / "proper_scores" / "g4_k8"
        self.output = self.root / "metric_comparisons" / "proper_scores" / "g4_k8"
        self.spec = ContinuousNMRSpec(num_measurements=64, num_spectrum_bins=8)
        self.truth = generate_fixed_split(
            num_samples=4, spec=self.spec, master_seed=91, split=PROPER_SCORE_TEST_SPLIT,
        )
        self.truth["proper_score_provenance"] = {
            "protocol_version": PROPER_SCORE_PROTOCOL_VERSION,
            "split": PROPER_SCORE_TEST_SPLIT,
            "master_seed": 91,
            "num_conditions": 4,
            "specification_hash": self.spec.specification_hash,
        }
        self.truth_path = self.run / "test_conditions.pt"
        _save_cache(self.truth_path, self.truth)
        self.source_path, self.source = self._model("conditional_vae_z16")

    def _model(self, variant: str) -> tuple[Path, dict]:
        deterministic = variant.startswith("deterministic_regression")
        method = "deterministic_regression" if deterministic else "conditional_vae"
        count = 1 if deterministic else 8
        if deterministic:
            samples = self.truth["histogram"].roll(2, dims=-1).unsqueeze(1)
        else:
            generator = torch.Generator().manual_seed(113)
            samples = torch.rand((4, count, 8), generator=generator)
            samples /= samples.sum(-1, keepdim=True)
        path = self.run / variant / "samples.pt"
        truth_hash = file_sha256(self.truth_path)
        provenance = {
            "protocol_version": PROPER_SCORE_PROTOCOL_VERSION,
            "variant": variant, "model_name": variant, "method": method, "nfe": 1,
            "conditions_sha256": truth_hash,
            "requested_sample_count": 8, "stored_sample_count": count,
            "master_source_seed": 123,
            "derived_source_seed": _derived_seed(123, PROPER_SCORE_PROTOCOL_VERSION, variant, 1, truth_hash),
            "checkpoint_identity": {"weights": "fixture"}, "model_subfolder": "unet_ema",
            "world_size": 1, "eval_batch_size": 4,
            "mixed_precision": "no", "max_decode_batch_size": 32,
        }
        artifact = {
            "protocol_id": self.spec.protocol_id,
            "specification": self.spec.to_dict(),
            "specification_hash": self.spec.specification_hash,
            "condition_ids": self.truth["sample_id"].clone(),
            "measurements": self.truth["measurement"].clone(),
            "grid": self.truth["grid"], "samples": samples,
            "deterministic": deterministic, "method": method,
            "training_config": {"latent_dim": 16, "regression_loss": "mse"},
            "sampling_config": {}, "source_seed": 123,
            "proper_score_provenance": provenance,
        }
        _save_cache(path, artifact)
        return path, artifact

    def _args(self, *extra: str, plots: bool = False) -> list[str]:
        args = ["--experiment_dir", str(self.root), "--num_conditions", "4",
                "--sample_count", "8", "--kernel_bandwidth", "0.4",
                "--num_cpu_threads", "1", "--ground_metrics", "l2"]
        if not plots:
            args.append("--skip_plots")
        return args + list(extra)

    def _evaluate(self, *extra: str, plots: bool = False) -> dict:
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("Offline scores must not run a network")):
            return evaluation.main(self._args(*extra, plots=plots))

    def _evaluate_default_metrics(self, *extra: str, plots: bool = False) -> dict:
        args = self._args(plots=plots)
        explicit_metric = args.index("--ground_metrics")
        del args[explicit_metric:explicit_metric + 2]
        args.extend(["--kernel_bandwidth_w2", "0.7", *extra])
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("Offline scores must not run a network")):
            return evaluation.main(args)

    @staticmethod
    def _manual_scores(samples: torch.Tensor, truth: torch.Tensor, deterministic: bool) -> tuple[float, float]:
        samples, truth = samples.double(), truth.double()
        samples = samples / samples.sum(-1, keepdim=True)
        truth = truth / truth.sum()
        truth_distance = torch.linalg.vector_norm(samples - truth, dim=-1)
        if deterministic:
            spread, within_kernel = 0.0, 1.0
        else:
            pair_distance = torch.cdist(samples, samples)
            off_diagonal = ~torch.eye(len(samples), dtype=torch.bool)
            spread = float(pair_distance[off_diagonal].mean())
            within_kernel = float(torch.exp(-pair_distance[off_diagonal].square() / (2 * 0.4**2)).mean())
        energy = float(truth_distance.mean()) - 0.5 * spread
        kernel = within_kernel - 2 * float(torch.exp(-truth_distance.square() / (2 * 0.4**2)).mean())
        return energy, kernel

    def _manual_w2_scores(self, samples: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
        edges = self.truth["grid"]["log_rate_edges"]
        truth_distance = pairwise_w2_exact(samples, truth[None, :], edges).flatten()
        pair_distance = pairwise_w2_exact(samples, samples, edges)
        off_diagonal = ~torch.eye(len(samples), dtype=torch.bool)
        spread = pair_distance[off_diagonal].mean()
        within_kernel = torch.exp(-pair_distance[off_diagonal].square() / (2 * 0.7**2)).mean()
        energy = truth_distance.mean() - 0.5 * spread
        kernel = within_kernel - 2 * torch.exp(-truth_distance.square() / (2 * 0.7**2)).mean()
        return float(energy), float(kernel)

    def test_default_metrics_publish_four_correct_summaries(self) -> None:
        result = self._evaluate_default_metrics()
        self.assertEqual(set(result), {"energy_score", "kernel_score", "energy_score_w2", "kernel_score_w2"})
        for ground_metric in ("l2", "w2"):
            for score_index, score_name in enumerate(("energy_score", "kernel_score")):
                with self.subTest(ground_metric=ground_metric, score_name=score_name):
                    key = score_name if ground_metric == "l2" else f"{score_name}_w2"
                    summary = result[key]
                    self.assertEqual(summary["ground_metric"], ground_metric)
                    self.assertEqual(summary["computed_count"], 1)
                    self.assertEqual(summary["condition_count"], 4)
                    self.assertFalse(summary["network_forward_performed"])
                    self.assertFalse(summary["smc_posterior_samples_used"])
                    report = summary["records"][0]["report"]
                    self.assertEqual(report["ground_metric"], ground_metric)
                    self.assertEqual(report["metric"], ground_metric)
                    if ground_metric == "w2":
                        self.assertEqual(report["distance_protocol_version"], W2_EXACT_PROTOCOL_VERSION)
                        self.assertFalse(report["quantile_quadrature_used"])
                        expected = [self._manual_w2_scores(samples, truth)[score_index]
                                    for samples, truth in zip(self.source["samples"], self.truth["histogram"])]
                    else:
                        expected = [self._manual_scores(samples, truth, False)[score_index]
                                    for samples, truth in zip(self.source["samples"], self.truth["histogram"])]
                    if score_name == "kernel_score":
                        self.assertEqual(report["kernel_bandwidth"], 0.4 if ground_metric == "l2" else 0.7)
                    for row, value in zip(report["per_condition"], expected):
                        self.assertAlmostEqual(row[score_name], value, places=12)
                    self.assertAlmostEqual(report["aggregate"][score_name]["mean"], sum(expected) / 4, places=12)
                    self.assertEqual(report["source_sha256"]["simulator_truth"], file_sha256(self.truth_path))
                    self.assertEqual(report["source_sha256"]["samples"], file_sha256(self.source_path))
                    self.assertEqual(json.loads((self.output / f"{key}_summary.json").read_text()), summary)
                    with (self.output / f"{key}_summary.csv").open() as handle:
                        self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_default_upgrade_keeps_legacy_l2_reports_and_only_scores_w2(self) -> None:
        self._evaluate()
        model_output = self.output / "models" / "conditional_vae_z16"
        preserved = {}
        for name in ("energy_score", "kernel_score"):
            path = model_output / f"{name}.json"
            report = json.loads(path.read_text())
            # Model a pre-extension report: the L2 protocol and formulas still
            # identify its metric without the new explicit distance metadata.
            for field in ("metric", "ground_metric", "distance_protocol_version",
                          "distance_computation", "distance_squared", "quantile_quadrature_used"):
                report.pop(field, None)
                report.get("bandwidth_provenance", {}).pop(field, None)
            path.write_text(json.dumps(report))
            preserved[path] = path.read_bytes()
        with patch.object(evaluation, "compute_energy_score_report", wraps=evaluation.compute_energy_score_report) as energy, \
                patch.object(evaluation, "compute_kernel_score_report", wraps=evaluation.compute_kernel_score_report) as kernel:
            result = self._evaluate_default_metrics()
        for scorer in (energy, kernel):
            self.assertEqual([call.kwargs["metric"] for call in scorer.call_args_list], ["w2"])
        for path, content in preserved.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(len(result), 4)
        self.assertTrue((model_output / "energy_score_w2.json").is_file())
        self.assertTrue((model_output / "kernel_score_w2.json").is_file())

    def test_recompute_refreshes_both_default_ground_metrics(self) -> None:
        first = self._evaluate_default_metrics()
        with patch.object(evaluation, "compute_energy_score_report", wraps=evaluation.compute_energy_score_report) as energy, \
                patch.object(evaluation, "compute_kernel_score_report", wraps=evaluation.compute_kernel_score_report) as kernel:
            repeated = self._evaluate_default_metrics("--recompute")
        for scorer in (energy, kernel):
            self.assertEqual([call.kwargs["metric"] for call in scorer.call_args_list], ["l2", "w2"])
        self.assertEqual(first, repeated)

    def test_recompute_reuses_persisted_independent_bandwidth_after_memory_cache_reset(self) -> None:
        args = self._args()
        for option in ("--ground_metrics", "--kernel_bandwidth"):
            index = args.index(option)
            del args[index:index + 2]
        kernel_metrics._calibrated_bandwidth.cache_clear()
        self.addCleanup(kernel_metrics._calibrated_bandwidth.cache_clear)
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("Offline scores must not run a network")), \
                patch.object(kernel_metrics, "simulate_from_key", wraps=kernel_metrics.simulate_from_key) as simulate:
            first = evaluation.main(args)
        self.assertEqual(simulate.call_count, kernel_metrics.KERNEL_SCORE_CALIBRATION_COUNT)
        calibration_paths = sorted((self.output / "calibration").rglob("kernel_bandwidth_*.json"))
        self.assertEqual(len(calibration_paths), 2)
        persisted = {path: path.read_bytes() for path in calibration_paths}
        calibration = {json.loads(content)["ground_metric"]: json.loads(content) for content in persisted.values()}
        self.assertNotEqual(calibration["l2"]["kernel_bandwidth"], calibration["w2"]["kernel_bandwidth"])
        self.assertEqual(
            calibration["l2"]["bandwidth_provenance"]["calibration_histogram_sha256"],
            calibration["w2"]["bandwidth_provenance"]["calibration_histogram_sha256"],
        )
        # Erase every in-process calibration cache to exercise the same disk
        # provenance path required by a later independent evaluator process.
        kernel_metrics._calibrated_bandwidth.cache_clear()
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("Offline scores must not run a network")), \
                patch.object(kernel_metrics, "_calibrated_bandwidth", side_effect=AssertionError("Persisted bandwidth must be reused")), \
                patch.object(kernel_metrics, "simulate_from_key", side_effect=AssertionError("Cached evaluation must not simulate")):
            repeated = evaluation.main([*args, "--recompute"])
        self.assertEqual(first, repeated)
        for path, content in persisted.items():
            self.assertEqual(path.read_bytes(), content)
        for ground_metric, cache in calibration.items():
            key = "kernel_score" if ground_metric == "l2" else "kernel_score_w2"
            report = repeated[key]["records"][0]["report"]
            self.assertEqual(report["kernel_bandwidth"], cache["kernel_bandwidth"])
            self.assertEqual(report["bandwidth_provenance"], cache["bandwidth_provenance"])

    def test_stale_w2_score_protocol_is_rejected_before_cache_reuse_or_plotting(self) -> None:
        self._evaluate_default_metrics()
        path = self.output / "models" / "conditional_vae_z16" / "energy_score_w2.json"
        stale = json.loads(path.read_text())
        stale["protocol_version"] = "truth_histogram_w2_quadrature_energy_score_u_v0"
        path.write_text(json.dumps(stale))
        saved_summary = (self.output / "energy_score_w2_summary.json").read_bytes()
        with patch.object(evaluation, "compute_energy_score_report", side_effect=AssertionError("Stale protocol must be rejected before scoring")), \
                patch.object(evaluation, "compute_kernel_score_report", side_effect=AssertionError("Stale protocol must be rejected before scoring")):
            for extra in ((), ("--plots_only",)):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "energy_score_w2.json"):
                    self._evaluate_default_metrics(*extra)
        self.assertEqual((self.output / "energy_score_w2_summary.json").read_bytes(), saved_summary)

    def test_plots_only_renders_both_cached_metrics_without_scoring(self) -> None:
        first = self._evaluate_default_metrics()
        with patch.object(evaluation, "compute_energy_score_report", side_effect=AssertionError("plots_only ES")), \
                patch.object(evaluation, "compute_kernel_score_report", side_effect=AssertionError("plots_only KS")):
            repeated = self._evaluate_default_metrics("--plots_only", plots=True)
        self.assertEqual(first, repeated)
        for ground_metric in ("l2", "w2"):
            suffix = "" if ground_metric == "l2" else "_w2"
            for extension in ("png", "pdf"):
                self.assertTrue((self.output / "figures" / f"proper_scores{suffix}_comparison.{extension}").is_file())
                for score in ("energy_score", "kernel_score"):
                    key = f"{score}{suffix}"
                    self.assertTrue((self.output / "figures" / key / f"{key}_comparison.{extension}").is_file())

    def test_plots_only_skips_wholly_absent_w2_without_scoring(self) -> None:
        first = self._evaluate()
        with patch.object(evaluation, "compute_energy_score_report", side_effect=AssertionError("plots_only ES")), \
                patch.object(evaluation, "compute_kernel_score_report", side_effect=AssertionError("plots_only KS")):
            repeated = self._evaluate_default_metrics("--plots_only")
        self.assertEqual(first, repeated)
        self.assertFalse((self.output / "energy_score_w2_summary.json").exists())
        self.assertFalse((self.output / "kernel_score_w2_summary.json").exists())

    def test_plots_only_rejects_partial_or_empty_score_reports_without_scoring(self) -> None:
        with patch.object(evaluation, "compute_energy_score_report", side_effect=AssertionError("plots_only ES")), \
                patch.object(evaluation, "compute_kernel_score_report", side_effect=AssertionError("plots_only KS")):
            with self.assertRaises(ValueError):
                self._evaluate_default_metrics("--plots_only")
        self._evaluate_default_metrics()
        (self.output / "models" / "conditional_vae_z16" / "kernel_score_w2.json").unlink()
        with patch.object(evaluation, "compute_energy_score_report", side_effect=AssertionError("plots_only ES")), \
                patch.object(evaluation, "compute_kernel_score_report", side_effect=AssertionError("plots_only KS")):
            with self.assertRaises(ValueError):
                self._evaluate_default_metrics("--plots_only")

    def test_parallel_model_scoring_publishes_both_models_without_lost_summary(self) -> None:
        self._model("deterministic_regression")
        program = """
import os
import time
from pathlib import Path
from codes.evaluate.evaluate_proper_scores import main
gate = Path(os.environ['SCORE_TEST_GATE'])
while not gate.exists():
    time.sleep(0.01)
main()
"""
        gate = self.root / "start"
        env = dict(os.environ, SCORE_TEST_GATE=str(gate), CUDA_VISIBLE_DEVICES="",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        processes = [subprocess.Popen(
            [sys.executable, "-c", program, *self._args("--variant", variant)],
            cwd=Path(__file__).resolve().parents[1], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ) for variant in ("conditional_vae_z16", "deterministic_regression")]
        gate.touch()
        try:
            for process in processes:
                output, _ = process.communicate(timeout=45)
                self.assertEqual(process.returncode, 0, output)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        for metric in ("energy_score", "kernel_score"):
            summary = json.loads((self.output / f"{metric}_summary.json").read_text())
            self.assertEqual(summary["computed_count"], 2)
        pairs = json.loads((self.output / "paired_score_differences.json").read_text())
        self.assertEqual(len(pairs["records"]), 2)

    def test_not_yet_published_other_model_samples_do_not_block_completed_model(self) -> None:
        pending = self.run / "other_model" / "samples.pt"
        pending.parent.mkdir()
        pending.write_bytes(b"not published yet")
        result = self._evaluate("--variant", "conditional_vae_z16")
        self.assertEqual(result["energy_score"]["computed_count"], 1)

    def test_both_scores_match_manual_u_statistics_without_network_forward(self) -> None:
        result = self._evaluate()
        for metric_index, metric in enumerate(("energy_score", "kernel_score")):
            summary = result[metric]
            self.assertEqual(summary["condition_count"], 4)
            self.assertEqual(summary["requested_sample_count"], 8)
            self.assertFalse(summary["network_forward_performed"])
            self.assertFalse(summary["smc_posterior_samples_used"])
            report = summary["records"][0]["report"]
            expected = [self._manual_scores(samples, truth, False)[metric_index]
                        for samples, truth in zip(self.source["samples"], self.truth["histogram"])]
            for row, value in zip(report["per_condition"], expected):
                self.assertAlmostEqual(row[metric], value, places=12)
            self.assertAlmostEqual(report["aggregate"][metric]["mean"], sum(expected)/4, places=12)
            self.assertEqual(report["source_sha256"]["simulator_truth"], file_sha256(self.truth_path))
            self.assertTrue((self.output / f"{metric}_summary.csv").is_file())

    def test_per_variant_runs_merge_both_scores_dirac_and_exact_paired_differences(self) -> None:
        self._model("deterministic_regression")
        first = self._evaluate("--variant", "conditional_vae_z16")
        self.assertEqual(first["energy_score"]["computed_count"], 1)
        result = self._evaluate("--variant", "deterministic_regression", plots=True)
        for suffix in ("png", "pdf"):
            self.assertTrue((self.output / "figures" / f"proper_scores_comparison.{suffix}").is_file())
        paired = json.loads((self.output / "paired_score_differences.json").read_text())["records"]
        self.assertEqual(len(paired), 2)
        for metric, summary in result.items():
            self.assertEqual(summary["computed_count"], 2)
            records = summary["records"]
            self.assertEqual([item["report"]["sample_count"] for item in records], [8, 1])
            validate_comparison_reports(records, score_name=metric)
            regression = records[1]["report"]
            self.assertTrue(regression["deterministic"])
            self.assertEqual(regression["requested_sample_count"], 8)
            for suffix in ("png", "pdf"):
                self.assertTrue((self.output / "figures" / metric / f"{metric}_comparison.{suffix}").is_file())
            expected = torch.tensor([a[metric] - b[metric] for a, b in zip(
                records[0]["report"]["per_condition"], records[1]["report"]["per_condition"],
            )], dtype=torch.float64)
            row = next(row for row in paired if row["metric"] == metric)
            self.assertAlmostEqual(row["mean_difference_left_minus_right"], float(expected.mean()), places=12)
            self.assertAlmostEqual(row["standard_error"], float(expected.std(correction=1))/math.sqrt(4), places=12)
            with (self.output / f"{metric}_summary.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_cached_rerun_performs_no_metric_recomputation(self) -> None:
        first = self._evaluate()
        with patch.object(evaluation, "compute_energy_score_report", side_effect=AssertionError("cached ES")), \
                patch.object(evaluation, "compute_kernel_score_report", side_effect=AssertionError("cached KS")):
            second = self._evaluate()
        self.assertEqual(first, second)

    def test_ids_observations_and_foreign_truth_hash_are_rejected(self) -> None:
        variants = []
        bad = copy.deepcopy(self.source)
        bad["condition_ids"] = bad["condition_ids"].roll(1)
        variants.append(bad)
        bad = copy.deepcopy(self.source)
        bad["measurements"][0, 0] += 1
        variants.append(bad)
        bad = copy.deepcopy(self.source)
        bad["proper_score_provenance"]["conditions_sha256"] = "foreign"
        variants.append(bad)
        for index, artifact in enumerate(variants):
            with self.subTest(case=index):
                _save_cache(self.source_path, artifact)
                with self.assertRaises(ValueError):
                    self._evaluate()
                self.assertFalse(list(self.output.rglob("energy_score.json")))
                self.assertFalse(list(self.output.rglob("kernel_score.json")))

    def test_wrong_g_k_or_unlabelled_generation_protocol_is_rejected(self) -> None:
        for changes in (
            {"protocol_version": "truth_score_sweep_generation_v1"},
            {"requested_sample_count": 4}, {"stored_sample_count": 4},
        ):
            artifact = copy.deepcopy(self.source)
            artifact["proper_score_provenance"].update(changes)
            _save_cache(self.source_path, artifact)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._evaluate()
        _save_cache(self.source_path, self.source)
        for flag, value in (("--num_conditions", "8"), ("--sample_count", "4")):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self._evaluate(flag, value, "--run_dir", str(self.run))
        self.assertFalse(self.output.exists())

    def test_legacy_and_pilot_truth_splits_are_rejected_before_scoring(self) -> None:
        for split in ("posterior_test", "score_sweep_test"):
            truth = copy.deepcopy(self.truth)
            truth["split"] = split
            truth["proper_score_provenance"]["split"] = split
            _save_cache(self.truth_path, truth)
            with self.subTest(split=split), self.assertRaises(ValueError):
                self._evaluate()
        self.assertFalse(self.output.exists())

    def test_truth_seed_must_match_declared_independent_provenance(self) -> None:
        self.truth["master_seed"] += 1
        _save_cache(self.truth_path, self.truth)
        self.source_path, self.source = self._model("conditional_vae_z16")
        with self.assertRaises(ValueError):
            self._evaluate()
        self.assertFalse(self.output.exists())

    def test_changed_generation_seed_cannot_reuse_previous_score_reports(self) -> None:
        self._evaluate()
        summary_path = self.output / "energy_score_summary.json"
        saved_summary = summary_path.read_bytes()
        self.source["proper_score_provenance"]["master_source_seed"] += 1
        _save_cache(self.source_path, self.source)
        with self.assertRaises(ValueError):
            self._evaluate()
        self.assertEqual(summary_path.read_bytes(), saved_summary)

    def test_corrupt_artifact_checksum_is_rejected_before_loading_or_scoring(self) -> None:
        with self.source_path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self._evaluate()
        self.assertFalse(self.output.exists())

    def test_changed_previously_scored_model_is_rejected_when_evaluating_another(self) -> None:
        self._model("deterministic_regression")
        self._evaluate("--variant", "conditional_vae_z16")
        self.source["samples"][0, 0] = self.source["samples"][0, 0].roll(1)
        _save_cache(self.source_path, self.source)
        with self.assertRaisesRegex(ValueError, "source changed"):
            self._evaluate("--variant", "deterministic_regression")

    def test_dirac_comparison_still_rejects_mismatched_truth_and_budget(self) -> None:
        self._model("deterministic_regression")
        result = self._evaluate()
        records = result["energy_score"]["records"]
        for field, value in (("requested_sample_count", 16), ("sample_count", 8)):
            modified = copy.deepcopy(records)
            modified[1]["report"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_comparison_reports(modified)
        modified = copy.deepcopy(records)
        modified[1]["report"]["source_sha256"]["simulator_truth"] = "different"
        with self.assertRaises(ValueError):
            validate_comparison_reports(modified)

    def test_legacy_experiment_scan_excludes_proper_score_and_diagnostic_banks(self) -> None:
        legacy_truth = generate_fixed_split(num_samples=4, spec=self.spec, master_seed=91, split="posterior_test")
        truth_path = self.root / "reference_test" / "test_conditions.pt"
        atomic_torch_save(truth_path, legacy_truth)
        sample = copy.deepcopy(self.source)
        sample["condition_ids"] = legacy_truth["sample_id"]
        sample["measurements"] = legacy_truth["measurement"]
        legacy_path = self.root / "old_model" / "evaluation" / "nested" / "samples.pt"
        atomic_torch_save(legacy_path, sample)
        # These intentionally invalid files must never be read by the legacy scan.
        pilot_path = self.root / "score_sweeps" / "variant" / "stage1" / "samples.pt"
        pilot_path.parent.mkdir(parents=True)
        pilot_path.write_bytes(b"not a legacy sample")
        legacy.main(["--experiment_dir", str(self.root), "--kernel_bandwidth", "0.4"])
        for metric in ("energy_score", "kernel_score"):
            path = self.root / "metric_comparisons" / metric / f"{metric}_summary.json"
            summary = json.loads(path.read_text())
            self.assertEqual(summary["computed_count"], 1)
            self.assertEqual(summary["records"][0]["samples"], str(legacy_path))
        self.assertFalse((self.source_path.parent / "metrics").exists())
        self.assertFalse((pilot_path.parent / "metrics").exists())


if __name__ == "__main__":
    unittest.main()
