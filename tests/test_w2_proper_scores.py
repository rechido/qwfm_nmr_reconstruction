import copy
import json
import math
import unittest
from unittest import mock

import torch

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.metrics.distances import W2_EXACT_PROTOCOL_VERSION, pairwise_w2_exact
from codes.metrics.energy_score import compute_energy_score_report, histogram_l2_energy_score
from codes.metrics.kernel_score import (
    _calibrated_bandwidth,
    compute_kernel_score_report,
    histogram_gaussian_kernel_score,
    simulate_from_key,
)


def artifacts():
    spec = ContinuousNMRSpec(num_spectrum_bins=8)
    grid = build_nmr_grid(spec)
    modes = torch.eye(8, dtype=torch.float64)
    common = {
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "grid": grid.as_artifact(),
    }
    samples = {
        **common,
        "condition_ids": torch.tensor([11, 22]),
        "measurements": torch.zeros(2, spec.num_measurements),
        "samples": torch.stack((modes[[0, 2, 2]], modes[[2, 4, 6]])),
    }
    truths = {
        **common,
        "sample_id": samples["condition_ids"].clone(),
        "measurement": samples["measurements"].clone(),
        "histogram": modes[[1, 5]],
    }
    return samples, truths


class W2ProperScoreTests(unittest.TestCase):
    def setUp(self):
        self.edges = torch.arange(4, dtype=torch.float64)
        self.samples = torch.eye(3, dtype=torch.float64)[[0, 2]]
        self.truth = torch.eye(3, dtype=torch.float64)[1]

    def test_energy_uses_unsquared_transport_and_distinct_pairs(self):
        with mock.patch("codes.metrics.distances.histograms_to_quantiles", side_effect=AssertionError("no quadrature")):
            result = histogram_l2_energy_score(
                self.samples, self.truth, metric="w2", log_rate_edges=self.edges,
                distance_chunk_size=1,
            )
        self.assertEqual(result["mean_distance_to_truth"], 1.0)
        self.assertEqual(result["mean_pairwise_distance"], 2.0)
        self.assertEqual(result["energy_score"], 0.0)
        self.assertEqual(result["metric"], "w2")

    def test_kernel_applies_gaussian_to_squared_exact_w2(self):
        result = histogram_gaussian_kernel_score(
            self.samples, self.truth, metric="w2", log_rate_edges=self.edges,
            bandwidth=0.5, distance_chunk_size=1,
        )
        self.assertAlmostEqual(result["mean_kernel_to_truth"], math.exp(-2))
        self.assertAlmostEqual(result["mean_pairwise_kernel"], math.exp(-8))
        self.assertAlmostEqual(result["kernel_score"], math.exp(-8) - 2 * math.exp(-2))

    def test_w2_point_forecasts_and_repeated_independent_values(self):
        point = self.samples[:1]
        kwargs = {"metric": "w2", "log_rate_edges": self.edges}
        for score, extra, pair_term, pair_value in (
            (histogram_l2_energy_score, {}, "mean_pairwise_distance", 0),
            (histogram_gaussian_kernel_score, {"bandwidth": 0.8}, "mean_pairwise_kernel", 1),
        ):
            with self.subTest(score=score.__name__):
                self.assertFalse(score(point, self.truth, **kwargs, **extra)["available"])
                single = score(point, self.truth, deterministic=True, **kwargs, **extra)
                repeated = score(point.repeat(4, 1), self.truth, **kwargs, **extra)
                self.assertEqual(single[pair_term], pair_value)
                self.assertEqual(repeated[pair_term], pair_value)
                term = "kernel_score" if extra else "energy_score"
                self.assertAlmostEqual(single[term], repeated[term])

    def test_metric_and_edges_are_explicit(self):
        for score, extra in (
            (histogram_l2_energy_score, {}),
            (histogram_gaussian_kernel_score, {"bandwidth": 1}),
        ):
            with self.subTest(score=score.__name__):
                with self.assertRaisesRegex(ValueError, "requires log_rate_edges"):
                    score(self.samples, self.truth, metric="w2", **extra)
                with self.assertRaisesRegex(ValueError, "support metric"):
                    score(self.samples, self.truth, metric="w1", **extra)
                default = score(self.samples, self.truth, **extra)
                explicit = score(self.samples, self.truth, metric="l2", **extra)
                self.assertEqual(default, explicit)


class W2ProperScoreReportTests(unittest.TestCase):
    def test_report_derives_grid_and_matches_per_condition_exact_scores(self):
        samples, truths = artifacts()
        edges = samples["grid"]["log_rate_edges"]
        for report_fn, point_fn, extra, term in (
            (compute_energy_score_report, histogram_l2_energy_score, {}, "energy_score"),
            (compute_kernel_score_report, histogram_gaussian_kernel_score, {"bandwidth": 0.7}, "kernel_score"),
        ):
            with self.subTest(term=term):
                result = report_fn(samples, truths, metric="w2", **extra)
                self.assertEqual(result["metric"], "w2")
                self.assertEqual(result["distance_protocol_version"], W2_EXACT_PROTOCOL_VERSION)
                self.assertEqual(result["log_rate_coordinate"], "natural_log_rate")
                self.assertFalse(result["distance_squared"])
                self.assertFalse(result["quantile_quadrature_used"])
                self.assertFalse(result["reference_posterior_samples_used"])
                for index, record in enumerate(result["per_condition"]):
                    expected = point_fn(
                        samples["samples"][index], truths["histogram"][index],
                        metric="w2", log_rate_edges=edges, **extra,
                    )
                    self.assertEqual(record[term], expected[term])
                no_grid_samples = {key: value for key, value in samples.items() if key != "grid"}
                no_grid_truths = {key: value for key, value in truths.items() if key != "grid"}
                self.assertEqual(result, report_fn(no_grid_samples, no_grid_truths, metric="w2", **extra))
                json.dumps(result, allow_nan=False)

    def test_w2_report_rejects_conflicting_grid_evidence_before_calibration(self):
        for target in ("samples", "truth"):
            for field in ("log_rate_edges", "rate_edges", "kernel", "specification_hash"):
                with self.subTest(target=target, field=field):
                    samples, truths = copy.deepcopy(artifacts())
                    artifact = samples if target == "samples" else truths
                    # The fixture intentionally shares its original grid mapping.
                    artifact["grid"] = copy.deepcopy(artifact["grid"])
                    if field == "specification_hash":
                        artifact["grid"][field] = "incorrect"
                    else:
                        artifact["grid"][field].view(-1)[1] += 0.001
                    with mock.patch("codes.metrics.kernel_score._calibrated_bandwidth", side_effect=AssertionError("invalid grid must not calibrate")):
                        with self.assertRaisesRegex(ValueError, "grid"):
                            compute_kernel_score_report(samples, truths, metric="w2")
                    with self.assertRaisesRegex(ValueError, "grid"):
                        compute_energy_score_report(samples, truths, metric="w2")

    def test_bandwidth_provenance_must_match_metric_specification_and_independence(self):
        samples, truths = artifacts()
        for provenance in (
            {"metric": "l2"},
            {"metric": "w2", "calibration_specification_hash": "incorrect"},
            {"metric": "w2", "evaluation_truths_used": True},
            {"metric": "w2", "distance_protocol_version": "quadrature"},
        ):
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                compute_kernel_score_report(
                    samples, truths, metric="w2", bandwidth=0.7,
                    bandwidth_provenance=provenance,
                )
        provenance = {"metric": "w2", "source": {"path": "independent_calibration.json"}}
        with mock.patch("codes.metrics.kernel_score._calibrated_bandwidth", side_effect=AssertionError("cached value must not recalibrate")):
            report = compute_kernel_score_report(
                samples, truths, metric="w2", bandwidth=0.7,
                bandwidth_provenance=provenance,
            )
        report["bandwidth_provenance"]["source"]["path"] = "changed"
        self.assertEqual(provenance["source"]["path"], "independent_calibration.json")

    def test_independent_bank_is_shared_but_bandwidth_is_metric_specific(self):
        samples, truths = artifacts()
        canonical = json.dumps(samples["specification"], sort_keys=True, separators=(",", ":"))
        _calibrated_bandwidth.cache_clear()
        try:
            # A small replay verifies the unchanged calibration-key procedure
            # and distinct metrics without making a full experiment fixture.
            with mock.patch("codes.metrics.kernel_score.KERNEL_SCORE_CALIBRATION_COUNT", 6), mock.patch(
                "codes.metrics.kernel_score.simulate_from_key", wraps=simulate_from_key,
            ) as simulate:
                l2_bandwidth, l2_provenance = _calibrated_bandwidth(canonical)
                w2_bandwidth, w2_provenance = _calibrated_bandwidth(canonical, metric="w2")
                self.assertEqual(simulate.call_count, 6)
                histograms = torch.stack([simulate_from_key(*call.args, **call.kwargs)["histogram"] for call in simulate.call_args_list])
                distances = pairwise_w2_exact(histograms, histograms, samples["grid"]["log_rate_edges"])
                indices = torch.triu_indices(6, 6, offset=1)
                expected = torch.quantile(distances[indices[0], indices[1]], 0.5).item()
                self.assertEqual(w2_bandwidth, expected)
                self.assertNotEqual(l2_bandwidth, w2_bandwidth)
                self.assertEqual(l2_provenance["calibration_histogram_sha256"], w2_provenance["calibration_histogram_sha256"])
                self.assertEqual(w2_provenance["selection"], "median_all_distinct_unordered_pairwise_w2")
                self.assertFalse(w2_provenance["evaluation_truths_used"])
                cached = compute_kernel_score_report(
                    samples, truths, metric="w2", bandwidth=w2_bandwidth,
                    bandwidth_provenance=w2_provenance,
                )
                automatic = compute_kernel_score_report(samples, truths, metric="w2")
                self.assertEqual(cached, automatic)
                self.assertEqual(simulate.call_count, 6)
        finally:
            _calibrated_bandwidth.cache_clear()


if __name__ == "__main__":
    unittest.main()
