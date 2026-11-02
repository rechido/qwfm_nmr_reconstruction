"""Selected-model visualization must not depend on unfinished experiments."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from codes.evaluate import visualize_results as visual
from tests import test_paper_results as fixtures


class ResultsVisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixtures.PaperReportTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.destination, self.protocol = self.fixture.root, self.fixture.destination, self.fixture.protocol

    def keep_models(self, *models: str) -> None:
        retained: set[Path] = set()
        for key, path in self.fixture.paths.items():
            if key[0] in models:
                retained.add(path)
                if key[-1] == "fc":
                    retained.add(path.with_name("per_condition.csv"))
        for path in self.root.rglob("*"):
            if path.is_file() and path not in retained:
                path.unlink()

    def clone_setting(self, m: int, n: int) -> None:
        source = self.root / "m64n128"
        destination = self.root / f"m{m}n{n}"
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = str(path.relative_to(source)).replace("m64_n128", f"m{m}_n{n}").replace("l128", f"l{n}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                data = json.loads(path.read_text().replace("spec64", f"spec{m}").replace("l128", f"l{n}"))
                if "num_measurements" in data:
                    data.update(num_measurements=m, num_spectrum_bins=n)
                target.write_text(json.dumps(data))
            else:
                target.write_bytes(path.read_bytes())

    def test_single_flow_completes_without_any_other_model_or_setting(self) -> None:
        self.keep_models("QWFM")
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        manifest = visual.export_results(self.root, self.destination, self.protocol, models=("QWFM",))
        self.assertEqual(manifest["visualization_level"], "single_model")
        self.assertEqual(manifest["models"], ["QWFM"])
        self.assertEqual(manifest["metric_row_count"], 24)
        self.assertFalse(manifest["tensor_files_opened"])
        self.assertFalse(manifest["network_forward_performed"])
        self.assertFalse(manifest["forward_errors_recomputed"])
        self.assertTrue(all("quantile_wasserstein" in path for path in manifest["source_reports_sha256"]))
        for stem, *_ in visual.PLOT_GROUPS:
            for extension in ("png", "pdf"):
                self.assertGreater((self.destination / f"{stem}.{extension}").stat().st_size, 1000)
        with (self.destination / "all_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["nfe"] for row in rows}, {"1", "256"})
        self.assertEqual({row["metric"] for row in rows}, set(visual.LABELS))
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_single_nonflow_plots_without_inventing_nfe(self) -> None:
        self.keep_models("Regression")
        manifest = visual.export_results(self.root, self.destination, self.protocol, models=("Regression",))
        self.assertEqual(manifest["metric_row_count"], 12)
        with (self.destination / "all_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["nfe"] for row in rows}, {""})
        import matplotlib.pyplot as plt
        figure, ax = plt.subplots()
        try:
            metric_rows, _ = visual.collect_reports(self.root, self.protocol, models=("Regression",))
            selected = [row for row in metric_rows if row["metric"] == "precision" and row["distance"] == "w2"]
            visual._draw_panel(ax, selected, ("Regression",), self.protocol.nfes)
            self.assertEqual(ax.get_xscale(), "linear")
            self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ["Regression"])
            self.assertIn("NFE independent", ax.get_xlabel())
        finally:
            plt.close(figure)

    def test_setting_comparison_and_all_settings_summary(self) -> None:
        one = visual.export_results(self.root, self.destination / "one", self.protocol, plots=False)
        self.assertEqual(one["visualization_level"], "setting_comparison")
        self.assertEqual(one["metric_row_count"], 72)
        self.assertEqual(one["settings"], [[64, 128]])
        self.clone_setting(32, 128)
        self.clone_setting(128, 256)
        protocol = replace(self.protocol, settings=visual.SETTINGS)
        # Exercise the multi-column renderer once; all six chart families are
        # already rendered independently in the single-model checks above.
        with patch.object(visual, "PLOT_GROUPS", visual.PLOT_GROUPS[:1]):
            combined = visual.export_results(self.root, self.destination / "all", protocol)
        self.assertEqual(combined["visualization_level"], "multiple_settings")
        self.assertEqual(combined["metric_row_count"], 216)
        with (self.destination / "all/metrics_at_max_nfe.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["setting"] for row in rows}, {"m32n128", "m64n128", "m128n256"})
        self.assertEqual(len(rows), 180)
        f1 = next(row for row in rows if row["setting"] == "m128n256" and row["model"] == "QWFM"
                  and row["metric"] == "f1" and row["distance"] == "w2")
        p, r = .4, 1.1 / 3
        self.assertAlmostEqual(float(f1["mean"]), 2 * p * r / (p + r), places=14)
        self.assertEqual(f1["standard_error"], "")
        self.assertEqual({row["nfe"] for row in rows if row["model"] in visual.FLOW_MODELS}, {"256"})
        self.assertFalse(any(row["distance"] == "w1" and row["metric"] in ("energy_score", "kernel_score") for row in rows))

    def test_missing_selected_report_fails_before_writing_outputs(self) -> None:
        self.keep_models("HFM")
        path = self.fixture.paths["HFM", 256, "fc"]
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "forward_consistency.json"):
            visual.export_results(self.root, self.destination, self.protocol, models=("HFM",))
        self.assertFalse(self.destination.exists())

    def test_model_subset_preserves_decoder_and_identity_checks(self) -> None:
        self.keep_models("QWFM")
        path = self.fixture.paths["QWFM", 1, "sampling"]
        original = path.read_bytes()
        self.fixture._edit(("QWFM", 1, "sampling"), lambda record: record["decoder_comparison"].update(decoder_name="hard_bin"))
        with self.assertRaisesRegex(ValueError, "QWFM PR decoder"):
            visual.export_results(self.root, self.destination, self.protocol, models=("QWFM",), plots=False)
        self.assertFalse(self.destination.exists())
        path.write_bytes(original)
        self.fixture._edit(("QWFM", 1, "kernel_score", "w2"), lambda record: record["source_sha256"].update(samples="wrong-draws"))
        with self.assertRaisesRegex(ValueError, "shared saved predictions"):
            visual.export_results(self.root, self.destination, self.protocol, models=("QWFM",), plots=False)
        self.assertFalse(self.destination.exists())

    def test_cli_selection_and_direct_launch(self) -> None:
        args = visual.parse_args(["--settings", "m128n256", "--models", "CVAE", "--nfes", "256", "4",
                                  "--output_dir", "outputs/m128n256/visualizations/cvae"])
        self.assertEqual(args.settings, ((128, 256),))
        self.assertEqual(args.models, ("CVAE",))
        self.assertEqual(args.nfes, (4, 256))
        script = Path(visual.__file__).resolve()
        result = subprocess.run([sys.executable, "-I", str(script), "--help"], cwd=self.root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--models", result.stdout)
        for arguments in (("--models", "HFM", "HFM"), ("--nfes", "0")):
            with self.assertRaises(SystemExit):
                visual.parse_args(["--settings", "m64n128", "--output_dir", "unused", *arguments])


if __name__ == "__main__":
    unittest.main()
