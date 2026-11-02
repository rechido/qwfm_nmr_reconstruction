import copy
import json
import math
import random
import unittest
from unittest import mock

import numpy as np
import torch

from codes.data.spec import ContinuousNMRSpec
from codes.metrics.kernel_score import (
    KERNEL_SCORE_CALIBRATION_COUNT,
    KERNEL_SCORE_CALIBRATION_SEED,
    KERNEL_SCORE_CALIBRATION_SPLIT,
    _calibrated_bandwidth,
    compute_kernel_score_report,
    histogram_gaussian_kernel_score,
    simulate_from_key,
)


def artifacts():
    spec = ContinuousNMRSpec(num_spectrum_bins=8)
    truth = torch.zeros(3, 8, dtype=torch.float64)
    truth[:, :2] = 0.5
    points = truth.clone()
    points[:, :2] = torch.tensor([[0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
    common = {
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
    }
    samples = {
        **common,
        "condition_ids": torch.tensor([11, 22, 33]),
        "measurements": torch.zeros(3, 32),
        "samples": points[:, None].repeat(1, 4, 1),
    }
    truths = {
        **common,
        "sample_id": samples["condition_ids"].clone(),
        "measurement": samples["measurements"].clone(),
        "histogram": truth,
    }
    return samples, truths


class HistogramGaussianKernelScoreTests(unittest.TestCase):
    def test_manual_two_draws_exclude_self_pairs(self):
        samples = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        truth = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = histogram_gaussian_kernel_score(
            samples, truth, bandwidth=0.5, distance_chunk_size=1
        )
        self.assertAlmostEqual(result["mean_kernel_to_truth"], math.exp(-1.0))
        self.assertAlmostEqual(result["mean_pairwise_kernel"], math.exp(-4.0))
        self.assertAlmostEqual(result["kernel_score"], math.exp(-4.0) - 2 * math.exp(-1.0))

    def test_two_mode_population_risk_matches_analytic_strict_optimum(self):
        modes = torch.eye(2, dtype=torch.float64)
        bandwidth = 0.7
        rho = math.exp(-1.0 / bandwidth**2)
        optimal = -(1 + rho) / 2
        # Exactly integrate the K=2 estimator over iid draws and one truth.
        for q in (0.0, 0.2, 0.5, 0.8, 1.0):
            probabilities = (q, 1 - q)
            expectation = 0.0
            for left in range(2):
                for right in range(2):
                    for truth in range(2):
                        score = histogram_gaussian_kernel_score(
                            modes[[left, right]], modes[truth], bandwidth=bandwidth
                        )["kernel_score"]
                        expectation += 0.5 * probabilities[left] * probabilities[right] * score
            self.assertAlmostEqual(expectation, optimal + 2 * (1 - rho) * (q - 0.5)**2)

    def test_point_forecast_has_pairwise_kernel_one_including_k_one(self):
        point = torch.tensor([[1.0, 0.0]])
        truth = torch.tensor([0.0, 1.0])
        single = histogram_gaussian_kernel_score(
            point, truth, bandwidth=1.0, deterministic=True
        )
        repeated = histogram_gaussian_kernel_score(point.repeat(4, 1), truth, bandwidth=1.0)
        self.assertEqual(single["mean_pairwise_kernel"], 1.0)
        self.assertAlmostEqual(single["kernel_score"], 1 - 2 * math.exp(-1))
        self.assertEqual(single["kernel_score"], repeated["kernel_score"])
        perfect = histogram_gaussian_kernel_score(point, point[0], bandwidth=1, deterministic=True)
        self.assertEqual(perfect["kernel_score"], -1.0)
        unavailable = histogram_gaussian_kernel_score(point, truth, bandwidth=1.0)
        self.assertFalse(unavailable["available"])
        self.assertNotIn("kernel_score", unavailable)

    def test_invalid_bandwidths_and_extreme_finite_bandwidths(self):
        point = torch.tensor([[1.0, 0.0]])
        for bandwidth in (None, 0, -1, math.inf, -math.inf, math.nan, True, "1"):
            with self.subTest(bandwidth=bandwidth):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    histogram_gaussian_kernel_score(point, point[0], bandwidth=bandwidth)
        for bandwidth in (5.0e-324, 1.0e308):
            result = histogram_gaussian_kernel_score(
                point, point[0], bandwidth=bandwidth, deterministic=True
            )
            self.assertEqual(result["kernel_score"], -1.0)

    def test_invalid_histograms_and_false_point_declaration(self):
        truth = torch.tensor([0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "unit mass"):
            histogram_gaussian_kernel_score(torch.tensor([[1.0, 1.0]]), truth, bandwidth=1)
        with self.assertRaisesRegex(ValueError, "different sample rows"):
            histogram_gaussian_kernel_score(torch.eye(2), truth, bandwidth=1, deterministic=True)

    def test_float64_chunked_computation_matches_direct_formula(self):
        samples = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]], dtype=torch.float32)
        truth = torch.tensor([0.6, 0.4], dtype=torch.float32)
        work = samples.double() / samples.double().sum(-1, keepdim=True)
        target = truth.double() / truth.double().sum()
        first = torch.exp(-((work - target)**2).sum(-1) / (2 * 0.3**2)).mean().item()
        pairs = torch.exp(-((work[:, None] - work[None, :])**2).sum(-1) / (2 * 0.3**2))
        second = (pairs.sum().item() - 3.0) / 6.0
        for chunk in (1, 2, 10):
            result = histogram_gaussian_kernel_score(samples, truth, bandwidth=0.3, distance_chunk_size=chunk)
            self.assertAlmostEqual(result["mean_kernel_to_truth"], first, places=14)
            self.assertAlmostEqual(result["mean_pairwise_kernel"], second, places=14)


class KernelScoreReportTests(unittest.TestCase):
    def setUp(self):
        self.samples, self.truth = artifacts()

    def test_condition_aggregation_not_reference_repeats(self):
        result = compute_kernel_score_report(self.samples, self.truth, bandwidth=0.5)
        values = torch.tensor(
            [1 - 2 * math.exp(-distance**2 / 0.5) for distance in (0, math.sqrt(2)/4, math.sqrt(2)/2)],
            dtype=torch.float64,
        )
        aggregate = result["aggregate"]["kernel_score"]
        self.assertEqual(len(result["per_condition"]), 3)
        self.assertAlmostEqual(aggregate["mean"], values.mean().item())
        self.assertAlmostEqual(aggregate["standard_deviation"], values.std().item())
        self.assertAlmostEqual(aggregate["standard_error"], values.std().item() / math.sqrt(3))
        self.assertEqual(result["aggregate"]["mean_pairwise_kernel"]["mean"], 1)
        self.assertFalse(result["reference_posterior_samples_used"])
        self.assertEqual(result["kernel_bandwidth"], 0.5)
        json.dumps(result, allow_nan=False)

    def test_selected_condition_and_reference_container_without_posterior_bank(self):
        reference = {
            **self.truth,
            "condition_ids": self.truth["sample_id"],
            "measurements": self.truth["measurement"],
            "truth_histograms": self.truth["histogram"],
            "reference_histograms": "not a truth and must never be read",
        }
        kwargs = {"bandwidth": 0.5, "condition_indices": [2], "sample_count": 2}
        result = compute_kernel_score_report(self.samples, reference, **kwargs)
        simulator = compute_kernel_score_report(self.samples, self.truth, **kwargs)
        self.assertEqual(result["per_condition"], simulator["per_condition"])
        self.assertEqual(result["aggregate"], simulator["aggregate"])
        self.assertEqual(result["per_condition"][0]["condition_id"], 33)
        self.assertIsNone(result["aggregate"]["kernel_score"]["standard_error"])

    def test_identity_and_specification_guards(self):
        changed = copy.deepcopy(self.truth)
        changed["sample_id"] = changed["sample_id"].flip(0)
        with self.assertRaisesRegex(ValueError, "condition IDs differ"):
            compute_kernel_score_report(self.samples, changed, bandwidth=1)
        changed = copy.deepcopy(self.truth)
        changed["measurement"][0, 0] = 1e-3
        with self.assertRaisesRegex(ValueError, "measurements differ"):
            compute_kernel_score_report(self.samples, changed, bandwidth=1)
        changed = copy.deepcopy(self.truth)
        changed["specification_hash"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "specification hash"):
            compute_kernel_score_report(self.samples, changed, bandwidth=1)
        changed = copy.deepcopy(self.truth)
        different_spec = ContinuousNMRSpec(num_spectrum_bins=8, noise_standard_deviation=0.01)
        changed.update(specification=different_spec.to_dict(), specification_hash=different_spec.specification_hash)
        with self.assertRaisesRegex(ValueError, "protocol/specification hashes differ"):
            compute_kernel_score_report(self.samples, changed, bandwidth=1)

    def test_missing_truth_never_triggers_calibration(self):
        with mock.patch("codes.metrics.kernel_score._calibrated_bandwidth", side_effect=AssertionError("must not calibrate")):
            missing = compute_kernel_score_report(self.samples)
            self.assertFalse(missing["available"])
            self.assertIsNone(missing["kernel_bandwidth"])
            self.assertFalse(missing["bandwidth_provenance"]["calibration_performed"])
            no_histogram = dict(self.truth)
            del no_histogram["histogram"]
            self.assertFalse(compute_kernel_score_report(self.samples, no_histogram)["available"])

    def test_stochastic_k_one_and_explicit_point_forecast(self):
        unavailable = compute_kernel_score_report(self.samples, self.truth, bandwidth=1, sample_count=1)
        self.assertFalse(unavailable["available"])
        self.assertIn("K=1", unavailable["reason"])
        self.samples["sampling_config"] = {"dirac_replication_count": 4}
        point = compute_kernel_score_report(self.samples, self.truth, bandwidth=1, sample_count=1)
        self.assertTrue(point["available"])
        self.assertEqual(point["aggregate"]["mean_pairwise_kernel"]["mean"], 1.0)
        self.samples["samples"][0, 3] = self.samples["samples"][2, 0]
        with self.assertRaisesRegex(ValueError, "different sample rows"):
            compute_kernel_score_report(self.samples, self.truth, bandwidth=1, sample_count=1)

    def test_default_calibration_is_reproducible_private_and_evaluation_independent(self):
        _calibrated_bandwidth.cache_clear()
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with mock.patch("codes.metrics.kernel_score.simulate_from_key", wraps=simulate_from_key) as simulate:
            original = compute_kernel_score_report(self.samples, self.truth)
            self.assertEqual(simulate.call_count, KERNEL_SCORE_CALIBRATION_COUNT)
            for index, call in enumerate(simulate.call_args_list):
                self.assertEqual(call.args, (index,))
                self.assertEqual(call.kwargs["master_seed"], KERNEL_SCORE_CALIBRATION_SEED)
                self.assertEqual(call.kwargs["split"], KERNEL_SCORE_CALIBRATION_SPLIT)
                self.assertEqual(call.kwargs["epoch"], 0)
            changed_samples, changed_truth = artifacts()
            changed_samples["samples"] = self.truth["histogram"][:, None].repeat(1, 2, 1)
            changed_truth["histogram"] = self.samples["samples"][:, 0]
            changed = compute_kernel_score_report(changed_samples, changed_truth, condition_indices=[1])
            self.assertEqual(simulate.call_count, KERNEL_SCORE_CALIBRATION_COUNT)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        self.assertEqual(original["kernel_bandwidth"], changed["kernel_bandwidth"])
        self.assertEqual(original["bandwidth_provenance"], changed["bandwidth_provenance"])
        self.assertEqual(original["bandwidth_provenance"]["calibration_pair_count"], 512 * 511 // 2)
        self.assertEqual(len(original["bandwidth_provenance"]["calibration_histogram_sha256"]), 64)
        self.assertNotEqual(original["aggregate"], changed["aggregate"])
        # Clear the cache to verify reproducibility rather than cache identity.
        _calibrated_bandwidth.cache_clear()
        repeated = compute_kernel_score_report(self.samples, self.truth)
        self.assertEqual(original, repeated)
        # A caller must not be able to corrupt later reports through metadata.
        repeated["bandwidth_provenance"]["calibration_sample_ids"][0] = -100
        final = compute_kernel_score_report(self.samples, self.truth)
        self.assertEqual(final["bandwidth_provenance"]["calibration_sample_ids"], [0, 511])
        json.dumps(final, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
