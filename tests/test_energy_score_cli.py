from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from codes.data import ContinuousNMRSpec, generate_fixed_split
from codes.evaluate.evaluate_energy_score import evaluate_one, main
from codes.evaluate.plot_energy_score import validate_comparison_reports
from codes.utils.io_utils import atomic_torch_save, atomic_write_json


class EnergyScoreCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        spec = ContinuousNMRSpec(num_spectrum_bins=8)
        self.truth = generate_fixed_split(
            num_samples=2, spec=spec, master_seed=91, split="posterior_test"
        )
        histograms = self.truth["histogram"]
        self.samples = {
            "protocol_id": spec.protocol_id,
            "specification": spec.to_dict(),
            "specification_hash": spec.specification_hash,
            "condition_ids": self.truth["sample_id"],
            "measurements": self.truth["measurement"],
            "samples": torch.stack((histograms, histograms.roll(1, dims=-1)), dim=1),
            "deterministic": False,
            "method": "test",
            "grid": self.truth["grid"],
        }
        self.sample_path = self.root / "model" / "evaluation" / "samples.pt"
        self.truth_path = self.root / "reference_test" / "test_conditions.pt"
        atomic_torch_save(self.sample_path, self.samples)
        atomic_torch_save(self.truth_path, self.truth)

    def test_backfill_preserves_previous_metrics_and_original_backup(self) -> None:
        metrics_dir = self.sample_path.parent / "metrics"
        previous = {
            "specification_hash": self.samples["specification_hash"],
            "settings": {"sample_count": 2},
            "condition_indices": [1],
            "unrelated_metrics": {"precision": 0.75, "values": [1, 2]},
        }
        for filename in ("evaluation_raw.json", "evaluation_summary.json"):
            atomic_write_json(metrics_dir / filename, previous)
        result = evaluate_one(self.sample_path, update_existing_metrics=True)
        self.assertTrue(result["available"])
        self.assertEqual(result["report"]["condition_count"], 1)
        self.assertFalse(result["report"]["network_forward_performed"])
        self.assertFalse(result["report"]["smc_posterior_samples_used"])
        self.assertTrue(result["kernel_report"]["available"])
        self.assertEqual(result["kernel_report"]["condition_count"], 1)
        self.assertGreater(result["kernel_report"]["kernel_bandwidth"], 0)
        for filename in ("evaluation_raw", "evaluation_summary"):
            updated = json.loads((metrics_dir / f"{filename}.json").read_text())
            updated.pop("energy_score")
            updated.pop("kernel_score")
            self.assertEqual(updated, previous)
            backup = metrics_dir / f"{filename}.before_energy_score.json"
            self.assertEqual(json.loads(backup.read_text()), previous)
            kernel_backup = metrics_dir / f"{filename}.before_kernel_score.json"
            self.assertEqual(json.loads(kernel_backup.read_text()), previous)
        evaluate_one(self.sample_path, update_existing_metrics=True)
        self.assertEqual(
            json.loads(
                (
                    metrics_dir / "evaluation_summary.before_energy_score.json"
                ).read_text()
            ),
            previous,
        )

    def test_truth_measurement_mismatch_is_rejected(self) -> None:
        self.truth["measurement"][0, 0] += 1.0
        atomic_torch_save(self.truth_path, self.truth)
        with self.assertRaises(ValueError):
            evaluate_one(self.sample_path, truth_file=self.truth_path)
        self.assertFalse(
            (self.sample_path.parent / "metrics" / "energy_score.json").exists()
        )
        self.assertFalse(
            (self.sample_path.parent / "metrics" / "kernel_score.json").exists()
        )

    def test_inconsistent_metric_files_are_rejected_before_any_write(self) -> None:
        metrics_dir = self.sample_path.parent / "metrics"
        base = {
            "specification_hash": self.samples["specification_hash"],
            "settings": {"sample_count": 2},
            "condition_indices": [0, 1],
        }
        for mismatch in (
            {"specification_hash": "different"},
            {"settings": {"sample_count": 1}},
            {"condition_indices": [1]},
            {"source_sha256": {"samples": "different"}},
            {"source_paths": {"samples": str(self.root / "missing.pt")}},
            {"kernel_score": {"source_sha256": {"samples": "different"}}},
        ):
            with self.subTest(mismatch=mismatch):
                # Summary determines K/conditions. A conflicting raw file must
                # leave both originals and any pre-existing sidecar untouched.
                previous = {
                    "evaluation_raw.json": {**base, **mismatch},
                    "evaluation_summary.json": base,
                }
                for filename, payload in previous.items():
                    atomic_write_json(metrics_dir / filename, payload)
                with self.assertRaises(ValueError):
                    evaluate_one(self.sample_path, update_existing_metrics=True)
                for filename, payload in previous.items():
                    self.assertEqual(
                        json.loads((metrics_dir / filename).read_text()), payload
                    )
                self.assertFalse(list(metrics_dir.glob("*.before_energy_score.json")))
                self.assertFalse((metrics_dir / "energy_score.json").exists())
                self.assertFalse(list(metrics_dir.glob("*.before_kernel_score.json")))
                self.assertFalse((metrics_dir / "kernel_score.json").exists())

    def test_empty_experiment_reports_missing_samples_without_inference(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        main(["--experiment_dir", str(empty), "--num_cpu_threads", "1"])
        report = json.loads(
            (
                empty / "metric_comparisons" / "energy_score" / "energy_score_summary.json"
            ).read_text()
        )
        self.assertEqual(report["status"], "no_saved_samples")
        self.assertEqual(report["sample_artifact_count"], 0)
        self.assertFalse(report["network_forward_performed"])
        kernel_report = json.loads(
            (empty / "metric_comparisons" / "kernel_score" / "kernel_score_summary.json").read_text()
        )
        self.assertEqual(kernel_report["status"], "no_saved_samples")
        self.assertFalse(kernel_report["network_forward_performed"])

    def test_explicit_bandwidth_changes_only_kernel_score(self) -> None:
        first = evaluate_one(self.sample_path, kernel_bandwidth=0.1)
        second = evaluate_one(self.sample_path, kernel_bandwidth=0.5)
        self.assertEqual(first["report"], second["report"])
        self.assertNotEqual(
            first["kernel_report"]["aggregate"], second["kernel_report"]["aggregate"]
        )
        self.assertEqual(second["kernel_report"]["kernel_bandwidth"], 0.5)
        with self.assertRaisesRegex(ValueError, "kernel_bandwidth"):
            validate_comparison_reports(
                [{"report": first["kernel_report"]}, {"report": second["kernel_report"]}],
                score_name="kernel_score",
            )

    def test_batch_writes_both_scores_under_metric_comparisons(self) -> None:
        main(["--experiment_dir", str(self.root), "--kernel_bandwidth", "0.2"])
        for score_name in ("energy_score", "kernel_score"):
            folder = self.root / "metric_comparisons" / score_name
            summary = json.loads((folder / f"{score_name}_summary.json").read_text())
            self.assertEqual(summary["computed_count"], 1)
            self.assertIn(score_name, summary["records"][0]["report"]["aggregate"])
            self.assertTrue((folder / f"{score_name}_summary.csv").is_file())

    def test_invalid_bandwidth_is_rejected_before_outputs(self) -> None:
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                main(["--experiment_dir", str(self.root), "--kernel_bandwidth", value])
        self.assertFalse((self.root / "metric_comparisons").exists())

    def test_explicit_batch_output_keeps_default_sidecars_unchanged(self) -> None:
        evaluate_one(self.sample_path, kernel_bandwidth=0.2)
        original = (self.sample_path.parent / "metrics" / "kernel_score.json").read_bytes()
        separate = self.root / "metric_comparisons" / "sensitivity"
        main(["--experiment_dir", str(self.root), "--kernel_bandwidth", "0.5",
              "--output_dir", str(separate)])
        self.assertEqual(
            (self.sample_path.parent / "metrics" / "kernel_score.json").read_bytes(), original
        )
        payload = json.loads((separate / "kernel_score_summary.json").read_text())
        record = payload["records"][0]
        self.assertEqual(record["report"]["kernel_bandwidth"], 0.5)
        self.assertTrue(Path(record["report_path"]).is_relative_to(separate))


if __name__ == "__main__":
    unittest.main()
