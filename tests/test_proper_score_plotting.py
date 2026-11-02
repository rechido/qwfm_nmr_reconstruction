"""Ground-metric isolation and legacy compatibility for saved-score figures."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codes.evaluate import plot_energy_score as plotting


class ProperScorePlottingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(plotting.plt.close, "all")

    @staticmethod
    def _record(score: str, metric: str | None = "l2") -> dict:
        terms = (
            ("energy_score", "mean_distance_to_truth", "mean_pairwise_distance")
            if score == "energy_score" else
            ("kernel_score", "mean_kernel_to_truth", "mean_pairwise_kernel")
        )
        report = {
            "protocol_version": f"truth_{metric or 'l2'}_{score}_u_v1",
            "specification_hash": "same-spectrum-grid",
            "source_sha256": {"simulator_truth": "same-truth", "samples": "same-samples"},
            "sample_count": 3,
            "requested_sample_count": 3,
            "condition_count": 2,
            "deterministic": False,
            "aggregate": {term: {"mean": 0.25, "standard_error": 0.01} for term in terms},
            "per_condition": [{"condition_id": index} for index in (1, 2)],
        }
        if metric is not None:
            report["ground_metric"] = metric
        if score == "kernel_score":
            report.update(kernel="gaussian", kernel_bandwidth=0.4)
        return {"available": True, "method_label": "FM", "nfe": 4, "report": report}

    def _summary(self, score: str, metric: str | None = "l2") -> Path:
        suffix = "_w2" if metric == "w2" else ""
        path = self.root / f"{score}{suffix}_summary.json"
        record = self._record(score, metric)
        point = copy.deepcopy(record)
        point.update(method_label="Regression-MSE", nfe=None)
        point["report"].update(sample_count=1, deterministic=True)
        point["report"]["source_sha256"]["samples"] = "same-point-prediction"
        payload = {
            "protocol_version": f"independent_{score}{suffix}_batch_v1",
            "records": [record, point],
        }
        if metric is not None:
            payload["ground_metric"] = metric
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_legacy_l2_reports_and_explicit_l2_reports_are_compatible(self) -> None:
        records = [self._record("energy_score", None), self._record("energy_score", "l2")]
        plotting.validate_comparison_reports(records, ground_metric="l2")

    def test_mixed_ground_metrics_are_rejected_for_each_score(self) -> None:
        for score in ("energy_score", "kernel_score"):
            records = [self._record(score, "l2"), self._record(score, "w2")]
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "ground_metric"):
                plotting.validate_comparison_reports(records, score_name=score)

    def test_mixed_protocols_are_rejected_even_with_same_metric(self) -> None:
        records = [self._record("energy_score", "w2") for _ in range(2)]
        records[1]["report"]["protocol_version"] = "different-w2-definition"
        with self.assertRaisesRegex(ValueError, "protocol_version"):
            plotting.validate_comparison_reports(records)

    def test_w2_kernel_bandwidth_must_be_shared_and_finite(self) -> None:
        for bandwidth in (0.7, 0, float("nan"), float("inf"), True):
            records = [self._record("kernel_score", "w2") for _ in range(2)]
            records[1]["report"]["kernel_bandwidth"] = bandwidth
            with self.subTest(bandwidth=bandwidth), self.assertRaisesRegex(ValueError, "bandwidth"):
                plotting.validate_comparison_reports(records, score_name="kernel_score")

    def test_explicit_ground_metric_must_match_legacy_reports(self) -> None:
        summary = self._summary("energy_score", None)
        with self.assertRaisesRegex(ValueError, "ground_metric"):
            plotting.main(["--summary", str(summary), "--ground_metric", "w2"])
        self.assertFalse(list(self.root.glob("*.png")))

    def test_summary_ground_metric_must_match_reports(self) -> None:
        summary = self._summary("energy_score", "w2")
        payload = json.loads(summary.read_text())
        payload["ground_metric"] = "l2"
        summary.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "ground_metric"):
            plotting.main(["--summary", str(summary)])
        self.assertFalse(list(self.root.glob("*.png")))

    def test_combined_panels_cannot_mix_ground_metrics(self) -> None:
        energy = self._summary("energy_score", "l2")
        kernel = self._summary("kernel_score", "w2")
        with self.assertRaisesRegex(ValueError, "ground_metric"):
            plotting.main(["--summary", str(energy), "--kernel_summary", str(kernel),
                           "--score", "both"])
        self.assertFalse(list(self.root.glob("*.png")))

    def test_combined_panels_require_same_prediction_sources(self) -> None:
        energy = self._record("energy_score", "w2")
        kernel = self._record("kernel_score", "w2")
        kernel["report"]["source_sha256"]["samples"] = "different-predictions"
        with self.assertRaisesRegex(ValueError, "prediction sources"):
            plotting._plot_combined_scores([energy], [kernel], self.root)

    def test_metric_labels_and_component_diagnostics_are_preserved(self) -> None:
        for metric in ("l2", "w2"):
            for score in ("energy_score", "kernel_score"):
                summary = self._summary(score, metric)
                captured = []

                def capture(fig, output, name):
                    captured.append((fig, output, name))

                with self.subTest(metric=metric, score=score), patch.object(plotting, "_save_figure", capture):
                    plotting.main(["--summary", str(summary)])
                    fig, output, name = captured[0]
                    self.assertEqual(len(fig.axes), 3)
                    self.assertIn(metric.upper(), fig.axes[0].get_title())
                    self.assertIn(metric.upper(), fig.axes[0].get_ylabel())
                    if metric == "w2":
                        self.assertIn("log-rate", fig.axes[0].get_title())
                        self.assertEqual(name, f"{score}_w2_comparison")
                    else:
                        self.assertEqual(name, f"{score}_comparison")
                    self.assertEqual(output, self.root)
                    self.assertIn("diagnostic", fig.axes[1].get_title())
                    self.assertIn("diagnostic", fig.axes[2].get_title())
                    # Both an NFE series and a deterministic horizontal baseline survive.
                    self.assertEqual(set(fig.axes[0].get_legend_handles_labels()[1]),
                                     {"FM", "Regression-MSE"})
                    plotting.plt.close(fig)

    def test_combined_auto_selects_metric_specific_kernel_and_saves_both_formats(self) -> None:
        for metric in (None, "w2"):
            energy = self._summary("energy_score", metric)
            self._summary("kernel_score", metric)
            plotting.main(["--summary", str(energy), "--score", "both"])
        for stem in ("proper_scores_comparison", "proper_scores_w2_comparison"):
            for suffix in ("png", "pdf"):
                path = self.root / f"{stem}.{suffix}"
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
