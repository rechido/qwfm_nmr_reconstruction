import copy
import hashlib
import json
import math
import unittest

import torch

from codes.data.spec import ContinuousNMRSpec
from codes.data.grid import build_nmr_grid
from codes.evaluate.evaluate_saved_samples import (
    EvaluationSettings,
    evaluate_saved_samples_artifacts,
)
from codes.metrics.energy_score import (
    compute_energy_score_report,
    histogram_l2_energy_score,
)


class HistogramEnergyScoreTests(unittest.TestCase):
    def test_two_draws_use_unsquared_l2_and_distinct_pair_denominator(self):
        samples = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        truth = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = histogram_l2_energy_score(samples, truth, distance_chunk_size=1)
        self.assertAlmostEqual(result["mean_distance_to_truth"], math.sqrt(0.5))
        self.assertAlmostEqual(result["mean_pairwise_distance"], math.sqrt(2.0))
        self.assertAlmostEqual(result["energy_score"], 0.0)

    def test_three_draws_match_explicit_independent_pairs(self):
        samples = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        truth = torch.tensor([0.25, 0.75])
        result = histogram_l2_energy_score(samples, truth, distance_chunk_size=2)
        first = sum(math.dist(row.tolist(), truth.tolist()) for row in samples) / 3
        second = sum(
            math.dist(samples[i].tolist(), samples[j].tolist())
            for i in range(3) for j in range(3) if i != j
        ) / 6
        self.assertAlmostEqual(result["mean_distance_to_truth"], first)
        self.assertAlmostEqual(result["mean_pairwise_distance"], second)
        self.assertAlmostEqual(result["energy_score"], first - second / 2)

    def test_single_draw_requires_explicit_deterministic_forecast(self):
        sample = torch.tensor([[0.0, 1.0]])
        truth = torch.tensor([1.0, 0.0])
        unavailable = histogram_l2_energy_score(sample, truth)
        self.assertFalse(unavailable["available"])
        self.assertNotIn("energy_score", unavailable)
        point = histogram_l2_energy_score(sample, truth, deterministic=True)
        repeated = histogram_l2_energy_score(sample.repeat(5, 1), truth)
        self.assertEqual(point["mean_pairwise_distance"], 0.0)
        self.assertAlmostEqual(point["energy_score"], math.sqrt(2.0))
        self.assertEqual(point["energy_score"], repeated["energy_score"])

    def test_invalid_simplex_and_false_point_declaration_are_rejected(self):
        truth = torch.tensor([0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "unit mass"):
            histogram_l2_energy_score(torch.tensor([[1.0, 1.0]]), truth)
        with self.assertRaisesRegex(ValueError, "different sample rows"):
            histogram_l2_energy_score(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]), truth, deterministic=True
            )


class EnergyScoreReportTests(unittest.TestCase):
    def setUp(self):
        self.spec = ContinuousNMRSpec(num_spectrum_bins=8)
        truth = torch.zeros(3, 8, dtype=torch.float64)
        truth[:, :2] = 0.5
        points = truth.clone()
        points[:, :2] = torch.tensor([[0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
        common = {
            "protocol_id": self.spec.protocol_id,
            "specification": self.spec.to_dict(),
            "specification_hash": self.spec.specification_hash,
        }
        self.samples = {
            **common,
            "condition_ids": torch.tensor([11, 22, 33]),
            "measurements": torch.zeros(3, 32),
            "samples": points[:, None].repeat(1, 4, 1),
        }
        self.truth = {
            **common,
            "sample_id": self.samples["condition_ids"].clone(),
            "measurement": self.samples["measurements"].clone(),
            "histogram": truth,
        }

    def test_aggregates_independent_conditions_with_sample_sd_and_se(self):
        result = compute_energy_score_report(self.samples, self.truth)
        unit = math.sqrt(2.0) / 4
        self.assertEqual(len(result["per_condition"]), 3)
        self.assertEqual(result["sample_count"], 4)
        self.assertFalse(result["reference_posterior_samples_used"])
        aggregate = result["aggregate"]["energy_score"]
        self.assertAlmostEqual(aggregate["mean"], unit)
        self.assertAlmostEqual(aggregate["standard_deviation"], unit)
        self.assertAlmostEqual(aggregate["standard_error"], unit / math.sqrt(3))
        self.assertEqual(aggregate["condition_count"], 3)
        json.dumps(result, allow_nan=False)

    def test_reference_container_and_simulator_truth_are_equivalent(self):
        reference = {
            **self.truth,
            "condition_ids": self.truth["sample_id"],
            "measurements": self.truth["measurement"],
            "truth_histograms": self.truth["histogram"],
            # The ES implementation must never read this posterior bank.
            "reference_histograms": "intentionally not a tensor",
        }
        simulator = compute_energy_score_report(self.samples, self.truth)
        stored = compute_energy_score_report(self.samples, reference)
        self.assertEqual(simulator["per_condition"], stored["per_condition"])
        self.assertEqual(simulator["aggregate"], stored["aggregate"])

    def test_identity_checks_prevent_misaligned_truths(self):
        wrong = copy.deepcopy(self.truth)
        wrong["sample_id"] = wrong["sample_id"].flip(0)
        with self.assertRaisesRegex(ValueError, "condition IDs differ"):
            compute_energy_score_report(self.samples, wrong)
        wrong = copy.deepcopy(self.truth)
        wrong["measurement"][0, 0] += 1.0e-3
        with self.assertRaisesRegex(ValueError, "measurements differ"):
            compute_energy_score_report(self.samples, wrong)
        wrong = copy.deepcopy(self.truth)
        wrong["specification_hash"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "specification hash"):
            compute_energy_score_report(self.samples, wrong)
        wrong = copy.deepcopy(self.samples)
        wrong["measurements"] = wrong["measurements"][:, :-1]
        wrong_truth = copy.deepcopy(self.truth)
        wrong_truth["measurement"] = wrong["measurements"]
        with self.assertRaisesRegex(ValueError, "measurements differ"):
            compute_energy_score_report(wrong, wrong_truth)

    def test_selected_conditions_are_scored_once_and_single_g_has_no_fake_se(self):
        result = compute_energy_score_report(
            self.samples, self.truth, condition_indices=[2], sample_count=2
        )
        self.assertEqual(result["condition_count"], 1)
        self.assertEqual(result["per_condition"][0]["condition_id"], 33)
        self.assertIsNone(result["aggregate"]["energy_score"]["standard_deviation"])
        self.assertIsNone(result["aggregate"]["energy_score"]["standard_error"])
        json.dumps(result, allow_nan=False)

    def test_missing_truth_and_stochastic_k_one_are_explicitly_unavailable(self):
        result = compute_energy_score_report(self.samples, None)
        self.assertFalse(result["available"])
        self.assertEqual(result["aggregate"], {})
        self.assertIn("No simulator truth artifact", result["reason"])
        no_truth = dict(self.truth)
        del no_truth["histogram"]
        result = compute_energy_score_report(self.samples, no_truth)
        self.assertFalse(result["available"])
        self.assertEqual(result["aggregate"], {})
        result = compute_energy_score_report(self.samples, self.truth, sample_count=1)
        self.assertFalse(result["available"])
        self.assertIn("K=1", result["reason"])

    def test_classical_explicit_dirac_metadata_allows_k_one(self):
        samples = copy.deepcopy(self.samples)
        samples["sampling_config"] = {"dirac_replication_count": 4}
        result = compute_energy_score_report(samples, self.truth, sample_count=1)
        self.assertTrue(result["available"])
        self.assertTrue(result["deterministic"])
        self.assertEqual(result["aggregate"]["mean_pairwise_distance"]["mean"], 0.0)

    def test_point_declaration_checks_all_rows_before_sample_count_selection(self):
        samples = copy.deepcopy(self.samples)
        samples["deterministic"] = True
        samples["samples"][0, 3] = samples["samples"][2, 0]
        with self.assertRaisesRegex(ValueError, "different sample rows"):
            compute_energy_score_report(samples, self.truth, sample_count=1)

    def test_shared_evaluation_adds_report_once_independently_of_smc_repeats(self):
        grid = build_nmr_grid(self.spec).as_artifact()
        samples = {**self.samples, "grid": grid}
        reference = {
            **self.truth,
            "condition_ids": self.truth["sample_id"],
            "measurements": self.truth["measurement"],
            "truth_histograms": self.truth["histogram"],
            "reference_histograms": self.truth["histogram"][:, None].repeat(1, 8, 1),
            "reference_run_id": 0,
            "reference_master_seed": 123,
            "smc_settings": {"fixture": True},
            "grid": grid,
        }
        oracle = {**reference, "reference_run_id": 1}
        reports = []
        kernel_reports = []
        for repeats in (1, 3):
            raw, summary = evaluate_saved_samples_artifacts(
                samples, reference, oracle,
                settings=EvaluationSettings(
                    sample_count=4, reference_repeats=repeats, nearest_k=1,
                    w2_num_quantiles=8, distance_chunk_size=2,
                    kernel_bandwidth=0.2,
                ),
            )
            self.assertEqual(len(raw["records"]), 3 * repeats)
            self.assertEqual(len(raw["energy_score"]["per_condition"]), 3)
            self.assertEqual(raw["energy_score"], summary["energy_score"])
            self.assertNotIn("energy_score", raw["records"][0]["model"])
            reports.append(summary["energy_score"])
            self.assertEqual(raw["kernel_score"], summary["kernel_score"])
            self.assertEqual(len(raw["kernel_score"]["per_condition"]), 3)
            self.assertEqual(raw["kernel_score"]["kernel_bandwidth"], 0.2)
            self.assertNotIn("kernel_score", raw["records"][0]["model"])
            kernel_reports.append(summary["kernel_score"])
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(kernel_reports[0], kernel_reports[1])

    def test_unknown_protocol_does_not_score_supplied_placeholder_truth(self):
        specification = {**self.spec.to_dict(), "protocol_id": "unsupported_protocol"}
        encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"))
        common = {
            "protocol_id": specification["protocol_id"],
            "specification": specification,
            "specification_hash": hashlib.sha256(encoded.encode()).hexdigest(),
        }
        samples = {**self.samples, **common}
        truth = {**self.truth, **common, "truth_histograms": "fabricated placeholder"}
        for artifact in (truth, None):
            with self.assertRaisesRegex(ValueError, "synthetic protocol only"):
                compute_energy_score_report(samples, artifact)


if __name__ == "__main__":
    unittest.main()
