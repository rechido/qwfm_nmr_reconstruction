"""Paper exports use saved reports and reject mixed scientific protocols."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from codes.evaluate import export_paper_results as export


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def stats(values: list[float], se_name: str = "standard_error") -> dict:
    return {"mean": sum(values)/len(values), se_name: statistics.stdev(values)/math.sqrt(len(values))}


class PaperReportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/"outputs"
        self.destination = Path(temporary.name)/"paper"
        self.protocol = export.PaperProtocol(settings=((64, 128),), nfes=(1, 256), pr_conditions=3,
                                             reference_samples=4, reference_repeats=2, score_conditions=3,
                                             score_samples=2, calibration_count=3)
        self.paths: dict[tuple, Path] = {}
        self._fixtures()

    def _fixtures(self) -> None:
        p = self.protocol
        experiment = self.root/"m64n128"
        for model in export.MODELS:
            variant, training = export.model_identity(64, 128, model)
            for nfe in (p.nfes if model in ("HFM", "QWFM") else (None,)):
                tail = Path(f"nfe_{nfe:04d}") if nfe else Path()
                directory = Path(training)/tail
                base = experiment/"decoder_comparison/linear_cdf" if model == "QWFM" else experiment
                pr_dir = experiment/training/("decoder_comparison/linear_cdf" if model == "QWFM" else "evaluation")/tail
                per_condition = [{"condition_id": index, "aggregate": {"model": {"ground_metrics": {}}}} for index in range(3)]
                aggregate = {"model": {"ground_metrics": {}}}
                for distance in ("l2", "w1", "w2"):
                    aggregate["model"]["ground_metrics"][distance] = {}
                    for condition in per_condition:
                        condition["aggregate"]["model"]["ground_metrics"][distance] = {}
                    for metric, values in (("precision", [.2, .4, .6]), ("recall", [.7, .3, .1])):
                        aggregate["model"]["ground_metrics"][distance][metric] = stats(values)
                        for condition, value in zip(per_condition, values):
                            condition["aggregate"]["model"]["ground_metrics"][distance][metric] = {"mean": value}
                pr = {"protocol_version": "saved_samples_reference_oracle_v1", "metric_protocol_version": "continuous_posterior_metrics_v1",
                      "specification_hash": "spec64", "settings": {"sample_count": 4, "reference_repeats": 2, "nearest_k": 3,
                      "w2_num_quantiles": 2048, "seed": 8}, "primary_reference_run_id": 0, "oracle_reference_run_id": 1,
                      "per_condition": per_condition, "aggregate": aggregate}
                pr_path = pr_dir/"metrics/evaluation_summary.json"
                save_json(pr_path, pr)
                self.paths[model, nfe, "pr"] = pr_path
                sampling = {"checkpoint_iteration": 200000, "model_subfolder": "unet_ema", "condition_ids": [0, 1, 2]}
                if model == "QWFM":
                    sampling["decoder_comparison"] = {"decoder_name": "linear_cdf"}
                # The actual CVAE PR evaluator has no JSON sampling sidecar.
                if model != "CVAE":
                    save_json(pr_dir/"sampling_summary.json", sampling)
                self.paths[model, nfe, "sampling"] = pr_dir/"sampling_summary.json"
                generation = {"protocol_version": "proper_score_generation_v1", "variant": variant, "nfe": nfe if nfe else 1,
                              "model_subfolder": "unet_ema", "checkpoint_identity": {"sha256": "checkpoint-"+model},
                              "conditions_sha256": "truth", "decoder_signature": {"decoder": "linear_cdf"}}
                sample_hash = f"{model}-{nfe}-samples"
                draws = 1 if model == "Regression" else 2
                for distance in ("l2", "w2"):
                    for metric, values in (("energy_score", [.3, .4, .5]), ("kernel_score", [-.7, -.6, -.5])):
                        report = {"protocol_version": export.SCORE_PROTOCOLS[metric, distance], "available": True,
                                  "variant": variant, "nfe": nfe, "condition_count": 3, "sample_count": draws,
                                  "requested_sample_count": 2, "specification_hash": "spec64", "ground_metric": distance,
                                  "distance_computation": "exact_piecewise_uniform_inverse_cdf_integration",
                                  "source_sha256": {"samples": sample_hash, "simulator_truth": "truth"},
                                  "generation_provenance": generation, "per_condition": [{"condition_id": i, metric: x} for i, x in enumerate(values)],
                                  "aggregate": {metric: stats(values)}}
                        if metric == "kernel_score":
                            report.update(kernel_bandwidth=.4, bandwidth_provenance={"calibration_count": 3,
                                "calibration_split": "kernel_score_calibration", "evaluation_predictions_used": False,
                                "evaluation_truths_used": False, "calibration_histogram_sha256": "calibration"})
                        path = base/"metric_comparisons/proper_scores/g3_k2/models"/directory/f"{metric}{'_w2' if distance == 'w2' else ''}.json"
                        save_json(path, report)
                        self.paths[model, nfe, metric, distance] = path
                fc_path = base/"metric_comparisons/forward_consistency/g3_k2/models"/directory/"forward_consistency.json"
                fc_path.parent.mkdir(parents=True, exist_ok=True)
                mean_values = [3., 2., 1.] if model == "Regression" else [1., .5, .25]
                with fc_path.with_name("per_condition.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["condition_id", "clean_normalized_mse", "mean_spectrum_clean_normalized_mse"])
                    writer.writeheader()
                    writer.writerows({"condition_id": i, "clean_normalized_mse": value, "mean_spectrum_clean_normalized_mse": mean_values[i]}
                                     for i, value in enumerate([3., 2., 1.]))
                save_json(fc_path, {"protocol_version": "saved_joint_pairs_forward_errors_v1",
                    "mean_spectrum_protocol_version": "arithmetic_mean_spectrum_forward_errors_v1", "variant": variant, "nfe": nfe,
                    "num_measurements": 64, "num_spectrum_bins": 128, "condition_count": 3, "requested_sample_count": 2,
                    "sample_count": draws, "specification_hash": "spec64", "truth_sha256": "truth", "samples_sha256": sample_hash,
                    "generation_provenance": generation,
                    "output_sha256": {"per_condition.csv": hashlib.sha256(fc_path.with_name("per_condition.csv").read_bytes()).hexdigest()},
                    "aggregate": {"clean_normalized_mse": stats([3., 2., 1.], "condition_standard_error"),
                                  "mean_spectrum_clean_normalized_mse": stats(mean_values, "condition_standard_error")}})
                self.paths[model, nfe, "fc"] = fc_path

    def _edit(self, key: tuple, change) -> None:
        path = self.paths[key]
        report = json.loads(path.read_text())
        change(report)
        save_json(path, report)

    def test_export_reads_reports_only_and_writes_all_paper_outputs(self) -> None:
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        # No sample tensor or checkpoint files exist in this fixture.
        manifest = export.export_results(self.root, self.destination, self.protocol)
        self.assertEqual(manifest["metric_row_count"], 72)
        for name in ("figure2_metrics_nfe.png", "figure2_metrics_nfe.pdf", "table1_distance_sensitivity.csv",
                     "table2_forward_consistency.csv", "all_metrics.csv", "manifest.json"):
            self.assertTrue((self.destination/name).is_file(), name)
        self.assertFalse(manifest["tensor_files_opened"])
        self.assertFalse(manifest["forward_errors_recomputed"])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        with (self.destination/"table1_distance_sensitivity.csv").open() as stream:
            table = list(csv.DictReader(stream))
        f1 = next(row for row in table if row["model"] == "QWFM" and row["metric"] == "f1" and row["distance"] == "w2")
        p, r = .4, 1.1/3
        self.assertAlmostEqual(float(f1["mean"]), 2*p*r/(p+r), places=14)
        self.assertFalse(any(row["distance"] == "w1" and row["metric"] in ("energy_score", "kernel_score") for row in table))
        with (self.destination/"table2_forward_consistency.csv").open() as stream:
            forward = list(csv.DictReader(stream))
        self.assertEqual(len(forward), 6)
        qwfm = next(row for row in forward if row["model"] == "QWFM")
        self.assertEqual(float(qwfm["individual_fc"]), 2.)
        self.assertAlmostEqual(float(qwfm["mean_spectrum_fc"]), 1.75/3)

    def test_rejects_wrong_decoder_before_creating_outputs(self) -> None:
        self._edit(("QWFM", 1, "sampling"), lambda r: r["decoder_comparison"].update(decoder_name="hard_bin"))
        with self.assertRaisesRegex(ValueError, "QWFM PR decoder"):
            export.export_results(self.root, self.destination, self.protocol)
        self.assertFalse(self.destination.exists())

    def test_rejects_wrong_checkpoint_iteration(self) -> None:
        self._edit(("HFM", 1, "sampling"), lambda r: r.update(checkpoint_iteration=100000))
        with self.assertRaisesRegex(ValueError, "checkpoint iteration"):
            export.collect_reports(self.root, self.protocol)

    def test_rejects_mixed_sample_identity_and_test_conditions(self) -> None:
        key = ("HFM", 1, "kernel_score", "w2")
        original = self.paths[key].read_text()
        self._edit(key, lambda r: r["source_sha256"].update(samples="different-predictions"))
        with self.assertRaisesRegex(ValueError, "shared saved predictions"):
            export.collect_reports(self.root, self.protocol)
        self.paths[key].write_text(original)
        self._edit(key, lambda r: r["per_condition"][0].update(condition_id=99))
        with self.assertRaisesRegex(ValueError, "condition IDs"):
            export.collect_reports(self.root, self.protocol)

    def test_rejects_approximate_w2_proper_scores(self) -> None:
        self._edit(("HFM", 1, "energy_score", "w2"), lambda r: r.update(distance_computation="midpoint_quadrature"))
        with self.assertRaisesRegex(ValueError, "W2 computation"):
            export.collect_reports(self.root, self.protocol)

    def test_old_forward_reports_request_offline_recompute(self) -> None:
        self._edit(("HFM", 1, "fc"), lambda r: r.pop("mean_spectrum_protocol_version"))
        with self.assertRaisesRegex(ValueError, "--recompute"):
            export.collect_reports(self.root, self.protocol)

    def test_rejects_forward_table_that_violates_mean_error_bound(self) -> None:
        path = self.paths["HFM", 1, "fc"]
        csv_path = path.with_name("per_condition.csv")
        content = csv_path.read_text().replace("0,3.0,1.0", "0,3.0,30.0")
        csv_path.write_text(content)
        self._edit(("HFM", 1, "fc"), lambda r: r["output_sha256"].update(
            {"per_condition.csv": hashlib.sha256(csv_path.read_bytes()).hexdigest()}))
        with self.assertRaisesRegex(ValueError, "mean-spectrum squared error"):
            export.collect_reports(self.root, self.protocol)

    def test_missing_report_has_actionable_error(self) -> None:
        self.paths["HFM", 1, "pr"].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "offline evaluation"):
            export.collect_reports(self.root, self.protocol)


if __name__ == "__main__":
    unittest.main()
