import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.evaluate.audit_reference import (
    AuditSettings,
    AuditThresholds,
    _derived_comparison,
    _peak_count_comparison,
    _status,
    audit_reference_artifacts,
    main as audit_main,
    validate_reference_artifacts,
)


class ReferenceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ContinuousNMRSpec(num_spectrum_bins=8)
        self.grid = build_nmr_grid(self.spec)
        generator = torch.Generator().manual_seed(4431)
        groups, samples = 2, 12
        truth = torch.rand((groups, 8), generator=generator, dtype=torch.float64)
        truth /= truth.sum(dim=-1, keepdim=True)
        measurements = truth @ self.grid.kernel.T
        reference_samples = torch.rand(
            (groups, samples, 8), generator=generator, dtype=torch.float64
        )
        reference_samples /= reference_samples.sum(dim=-1, keepdim=True)
        probabilities = torch.tensor(
            [[0.20, 0.25, 0.25, 0.20, 0.10], [0.10, 0.20, 0.30, 0.25, 0.15]],
            dtype=torch.float64,
        )
        common = {
            "protocol_id": self.spec.protocol_id,
            "specification": self.spec.to_dict(),
            "specification_hash": self.spec.specification_hash,
            "reference_master_seed": 7788,
            "smc_settings": {"num_particles": 12, "test": True},
            "condition_ids": torch.tensor([8001, 8002]),
            "measurements": measurements,
            "truth_histograms": truth,
            "truth_component_counts": torch.tensor([2, 3]),
            "grid": self.grid.as_artifact(),
        }
        self.first = {
            **common,
            "reference_run_id": 0,
            "reference_histograms": reference_samples,
            "posterior_component_count_probability": probabilities,
        }
        self.second = copy.deepcopy(self.first)
        self.second["reference_run_id"] = 1
        # A permutation represents the same empirical posterior and exercises
        # independent deterministic subsampling in the audit.
        self.second["reference_histograms"] = torch.flip(reference_samples, dims=(1,))

    def test_audit_is_condition_wise_and_reports_all_diagnostics(self) -> None:
        settings = AuditSettings(
            sample_count=8, w2_num_quantiles=16, distance_chunk_size=4, seed=91
        )
        first_before = self.first["reference_histograms"].clone()
        second_before = self.second["reference_histograms"].clone()
        raw, summary = audit_reference_artifacts(
            [self.first, self.second],
            settings=settings,
            thresholds=AuditThresholds(),
        )
        self.assertEqual(len(raw["agreement_records"]), 2)
        self.assertEqual(
            {record["condition_id"] for record in raw["agreement_records"]},
            {8001, 8002},
        )
        record = raw["agreement_records"][0]
        self.assertEqual(set(record["ground_metrics"]), {"l2", "w1", "w2"})
        self.assertIn("rate_2_10", record["derived_quantities"])
        self.assertIn("absolute_difference", record["forward_diagnostics"])
        self.assertEqual(record["model_order_tv"], 0.0)
        suitability = raw["benchmark_suitability_records"][0]
        self.assertEqual(
            set(suitability["within_posterior_offdiagonal_spread"]),
            {"l2", "w1", "w2"},
        )
        self.assertEqual(
            set(suitability["quantile_round_trip_error"]), {"l2", "w1", "w2"}
        )
        self.assertGreaterEqual(suitability["w2_spread_to_round_trip_floor_ratio"], 0.0)
        self.assertIn(
            "normalized_standard_deviation",
            suitability["derived_quantity_posterior_spread"]["rate_2_10"],
        )
        self.assertIn(summary["overall_status"], {"pass", "warn", "fail"})
        self.assertIn(summary["agreement_status"], {"pass", "warn", "fail"})
        self.assertIn(summary["suitability_status"], {"pass", "warn", "fail"})
        self.assertEqual(len(summary["agreement"]["checks"]), 11)
        self.assertEqual(len(summary["benchmark_suitability"]["checks"]), 5)
        duplicate_diagnostics = suitability["reference_duplicate_diagnostics"]
        self.assertEqual(duplicate_diagnostics["unique_sample_fraction"], 1.0)
        self.assertEqual(duplicate_diagnostics["maximum_multiplicity"], 1)
        self.assertEqual(
            duplicate_diagnostics["kth_neighbour_zero_radius_fraction"], 0.0
        )
        self.assertEqual(record["first_full_bank_sample_count"], 12)
        self.assertEqual(record["distance_sample_count"], 8)
        # The two full banks are permutations. Independent distance subsets can
        # differ, but scalar statistics and morphology must remain identical.
        self.assertAlmostEqual(record["maximum_normalized_derived_difference"], 0.0)
        self.assertEqual(record["peak_count_tv"], 0.0)
        self.assertTrue(torch.equal(self.first["reference_histograms"], first_before))
        self.assertTrue(torch.equal(self.second["reference_histograms"], second_before))
        for difference in record["forward_diagnostics"]["absolute_difference"].values():
            self.assertAlmostEqual(difference, 0.0)
        split = suitability["same_bank_split_diagnostics"]
        self.assertFalse(split["used_as_acceptance_gate"])
        self.assertEqual(split["first_sample_count"] + split["second_sample_count"], 12)

    def test_integer_median_jump_does_not_fail_continuous_gate(self) -> None:
        # Two near-balanced PMFs straddle 50%: their medians differ by one
        # although only 2% of the probability mass moved.
        first_peaks = torch.tensor([1.0] * 51 + [2.0] * 49, dtype=torch.float64)
        second_peaks = torch.tensor([1.0] * 49 + [2.0] * 51, dtype=torch.float64)
        quantities = [
            {"rate_2_10": torch.zeros(100), "effective_peak_count": peaks}
            for peaks in (first_peaks, second_peaks)
        ]
        with patch(
            "codes.evaluate.audit_reference.derived_quantities", side_effect=quantities
        ):
            comparison, maximum_continuous = _derived_comparison(
                torch.empty(100, 8),
                torch.empty(100, 8),
                self.grid.log_rate_edges,
                self.spec,
            )
        peak = comparison["effective_peak_count"]
        self.assertEqual(peak["absolute_difference"]["median"], 1.0)
        self.assertEqual(peak["normalized_absolute_difference"]["median"], 0.2)
        self.assertFalse(peak["used_in_continuous_derived_gate"])
        self.assertEqual(maximum_continuous, 0.0)
        tv = peak["probability_mass_comparison"]["total_variation"]
        self.assertAlmostEqual(tv, 0.02)
        self.assertEqual(_status(tv, 0.05, 0.1), "pass")

    def test_peak_probability_shift_still_fails_own_gate(self) -> None:
        result = _peak_count_comparison(
            torch.tensor([2] * 70 + [4] * 30),
            torch.tensor([2] * 38 + [4] * 62),
        )
        self.assertEqual(result["support"], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(result["total_variation"], 0.32)
        self.assertAlmostEqual(result["maximum_cdf_difference"], 0.32)
        thresholds = AuditThresholds()
        self.assertEqual(
            _status(
                result["total_variation"],
                thresholds.peak_count_tv_pass,
                thresholds.peak_count_tv_warn,
            ),
            "fail",
        )
        first, second = copy.deepcopy(self.first), copy.deepcopy(self.second)
        two_peak = (
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
            / 2
        )
        four_peak = (
            torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
            / 4
        )
        for artifact, two_peak_count in ((first, 70), (second, 38)):
            bank = torch.cat(
                (
                    two_peak.repeat(two_peak_count, 1),
                    four_peak.repeat(100 - two_peak_count, 1),
                )
            )
            artifact["reference_histograms"] = bank.unsqueeze(0).repeat(2, 1, 1)
        _, summary = audit_reference_artifacts(
            [first, second],
            settings=AuditSettings(
                sample_count=8,
                w2_num_quantiles=16,
                distance_chunk_size=4,
                representation_quantiles=8,
            ),
            thresholds=thresholds,
        )
        checks = {check["name"]: check for check in summary["agreement"]["checks"]}
        self.assertAlmostEqual(checks["peak_count_tv_p95"]["value"], 0.32)
        self.assertEqual(checks["peak_count_tv_p95"]["status"], "fail")

    def test_duplicate_heavy_reference_fails_even_when_spread_gates_are_disabled(
        self,
    ) -> None:
        first = copy.deepcopy(self.first)
        second = copy.deepcopy(self.second)
        duplicate_cloud = self.first["reference_histograms"][:, :3].repeat_interleave(
            4, dim=1
        )
        first["reference_histograms"] = duplicate_cloud
        second["reference_histograms"] = duplicate_cloud.clone()
        permissive_spread_thresholds = AuditThresholds(
            minimum_w2_spread_pass=0.0,
            minimum_w2_spread_warn=0.0,
            minimum_w2_floor_ratio_pass=0.0,
            minimum_w2_floor_ratio_warn=0.0,
        )
        raw, summary = audit_reference_artifacts(
            [first, second],
            settings=AuditSettings(
                sample_count=8,
                w2_num_quantiles=16,
                distance_chunk_size=4,
                seed=25,
                representation_quantiles=8,
            ),
            thresholds=permissive_spread_thresholds,
        )
        diagnostics = raw["benchmark_suitability_records"][0][
            "reference_duplicate_diagnostics"
        ]
        self.assertEqual(diagnostics["unique_sample_fraction"], 0.25)
        self.assertEqual(diagnostics["maximum_multiplicity"], 4)
        self.assertEqual(diagnostics["kth_neighbour_zero_radius_fraction"], 1.0)
        checks = {
            check["name"]: check for check in summary["benchmark_suitability"]["checks"]
        }
        self.assertEqual(checks["reference_unique_fraction_minimum"]["status"], "fail")
        self.assertEqual(checks["reference_maximum_multiplicity"]["status"], "fail")
        self.assertEqual(
            checks["reference_k3_zero_radius_fraction_maximum"]["status"],
            "fail",
        )
        self.assertEqual(summary["suitability_status"], "fail")
        self.assertEqual(summary["overall_status"], "fail")

    def test_point_posterior_fails_suitability_but_not_agreement(self) -> None:
        point = self.first["truth_histograms"][:, None, :].repeat(1, 12, 1)
        first = copy.deepcopy(self.first)
        second = copy.deepcopy(self.second)
        first["reference_histograms"] = point
        second["reference_histograms"] = point.clone()
        _, summary = audit_reference_artifacts(
            [first, second],
            settings=AuditSettings(
                sample_count=8,
                w2_num_quantiles=16,
                distance_chunk_size=4,
                seed=14,
                representation_quantiles=8,
            ),
            thresholds=AuditThresholds(),
        )
        self.assertEqual(summary["agreement_status"], "pass")
        self.assertEqual(summary["suitability_status"], "fail")
        self.assertEqual(summary["overall_status"], "fail")
        suitability = summary["benchmark_suitability"]["aggregate"]
        self.assertEqual(suitability["w2_spread_mean_lower_quantile"], 0.0)

    def test_strict_identity_validation(self) -> None:
        with self.subTest("duplicate run id"):
            duplicate = copy.deepcopy(self.second)
            duplicate["reference_run_id"] = 0
            with self.assertRaisesRegex(ValueError, "distinct run IDs"):
                validate_reference_artifacts([self.first, duplicate], sample_count=8)
        with self.subTest("different settings"):
            mismatch = copy.deepcopy(self.second)
            mismatch["smc_settings"] = {"num_particles": 99}
            with self.assertRaisesRegex(ValueError, "different SMC settings"):
                validate_reference_artifacts([self.first, mismatch], sample_count=8)
        with self.subTest("condition identity"):
            mismatch = copy.deepcopy(self.second)
            mismatch["measurements"][0, 0] += 1.0e-4
            with self.assertRaisesRegex(ValueError, "measurements differ"):
                validate_reference_artifacts([self.first, mismatch], sample_count=8)
        with self.subTest("grid identity"):
            mismatch = copy.deepcopy(self.second)
            mismatch["grid"]["kernel"][0, 0] += 1.0e-4
            with self.assertRaisesRegex(ValueError, r"grid\['kernel'\]"):
                validate_reference_artifacts([self.first, mismatch], sample_count=8)

    def test_cli_writes_raw_and_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path, second_path = root / "run0.pt", root / "run1.pt"
            torch.save(self.first, first_path)
            torch.save(self.second, second_path)
            output = root / "audit"
            audit_main(
                [
                    "--references",
                    str(first_path),
                    str(second_path),
                    "--output_dir",
                    str(output),
                    "--sample_count",
                    "8",
                    "--w2_num_quantiles",
                    "16",
                    "--distance_chunk_size",
                    "4",
                    "--representation_quantiles",
                    "8",
                ]
            )
            for filename in (
                "reference_audit_raw.json",
                "reference_audit_summary.json",
            ):
                path = output / filename
                self.assertTrue(path.is_file())
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.assertEqual(
                    payload["specification_hash"], self.spec.specification_hash
                )


if __name__ == "__main__":
    unittest.main()
