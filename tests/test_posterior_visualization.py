import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
import codes.evaluate.visualize_posterior_samples as plotting


class PosteriorGalleryTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = ContinuousNMRSpec(num_spectrum_bins=8)
        grid = build_nmr_grid(spec).as_artifact()
        generator = torch.Generator().manual_seed(109)
        samples = torch.rand(1, 32, 8, generator=generator, dtype=torch.float64)
        samples /= samples.sum(dim=-1, keepdim=True)
        truth = samples[:, 0]
        common = {
            "protocol_id": spec.protocol_id,
            "specification": spec.to_dict(),
            "specification_hash": spec.specification_hash,
            "condition_ids": torch.tensor([701]),
            "measurements": truth @ grid["kernel"].T,
            "grid": grid,
        }
        self.reference = {
            **common,
            "truth_histograms": truth,
            "reference_histograms": samples,
        }
        self.model = {
            **common,
            "samples": samples[:, 16:],
            "method": "histogram_linear",
        }
        self.point = {
            **common,
            "samples": truth[:, None].expand(-1, 16, -1).clone(),
            "method": "deterministic_regression",
            "deterministic": True,
        }

    def test_galleries_preserve_header_kshot_and_unique_regression(self) -> None:
        captured = {}

        def capture(figure, path, *, dpi):
            del dpi
            captured[Path(path).name] = [
                {
                    "title": axis.get_title(),
                    "lines": len(axis.lines),
                    "x": axis.lines[0].get_xdata().copy(),
                    "y": axis.lines[0].get_ydata().copy(),
                    "ylim": axis.get_ylim(),
                }
                for axis in figure.axes
            ]
            Path(path).touch()

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(plotting, "_save_figure", side_effect=capture),
        ):
            manifest = plotting.visualize_posterior_artifacts(
                self.reference,
                {"HFM": self.model, "Regression-MSE": self.point},
                temporary,
                dpi=72,
            )
        self.assertEqual(
            set(captured), {"combined_k16.png", "HFM_k16.png", "Regression-MSE_k16.png"}
        )
        self.assertEqual(len(captured["combined_k16.png"]), 35)
        self.assertEqual(len(captured["HFM_k16.png"]), 34)
        self.assertEqual(len(captured["Regression-MSE_k16.png"]), 19)
        for axes in captured.values():
            self.assertEqual(axes[0]["title"], "Observed condition")
            self.assertIn("Simulator truth", axes[1]["title"])
            self.assertTrue(all(axis["lines"] == 1 for axis in axes))
            torch.testing.assert_close(
                torch.as_tensor(axes[0]["x"]),
                self.reference["grid"]["echo_times"] * 1000,
            )
            torch.testing.assert_close(
                torch.as_tensor(axes[1]["y"]), self.reference["truth_histograms"][0]
            )
        combined = captured["combined_k16.png"]
        for shot in range(16):
            torch.testing.assert_close(
                torch.as_tensor(combined[2 + shot]["y"]),
                self.reference["reference_histograms"][0, shot],
            )
            torch.testing.assert_close(
                torch.as_tensor(combined[18 + shot]["y"]),
                self.model["samples"][0, shot],
            )
        self.assertEqual(
            manifest["conditions"][0]["displayed_shots_by_model"],
            {"HFM": 16, "Regression-MSE": 1},
        )
        self.assertEqual(manifest["conditions"][0]["diagnostics"], {})

    def test_nfe_sweep_reuses_static_sheets_and_shared_axes(self) -> None:
        calls = []

        def capture(*args, **kwargs):
            del args
            calls.append(kwargs)
            kwargs["path"].touch()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(self.reference, root / "reference.pt")
            torch.save(self.point, root / "regression.pt")
            for nfe in (1, 4):
                directory = root / "hfm" / f"nfe_{nfe:04d}"
                directory.mkdir(parents=True)
                model = copy.deepcopy(self.model)
                if nfe == 4:
                    model["samples"][0, 0] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
                torch.save(model, directory / "samples.pt")
            output = root / "figures"
            with mock.patch.object(
                plotting, "_plot_kshot_gallery", side_effect=capture
            ):
                with self.assertWarnsRegex(UserWarning, "Skipping missing"):
                    plotting.main(
                        [
                            "--reference",
                            str(root / "reference.pt"),
                            "--models",
                            f"HFM={root / 'hfm/nfe_0004/samples.pt'}",
                            f"Regression-MSE={root / 'regression.pt'}",
                            f"CVAE-z16={root / 'not_trained.pt'}",
                            "--output_dir",
                            str(output),
                            "--nfe_values",
                            "1",
                            "4",
                            "--skip_missing",
                            "--dpi",
                            "72",
                        ]
                    )
            manifest = json.loads((output / "visualization_manifest.json").read_text())
            first, second = [run["conditions"][0] for run in manifest["runs"]]
            self.assertEqual(first["spectrum_ymax"], 1.05)
            self.assertEqual(second["spectrum_ymax"], 1.05)
            self.assertEqual(len(manifest["skipped_sources"]), 2)
            self.assertEqual(
                first["model_sheets_by_label"]["Regression-MSE"],
                second["model_sheets_by_label"]["Regression-MSE"],
            )
            self.assertIn(
                "nfe_independent", first["model_sheets_by_label"]["Regression-MSE"]
            )
            self.assertTrue(
                (output / "nfe_0001/condition_0000/combined_k16.png").is_file()
            )
            self.assertTrue(
                (output / "nfe_0004/condition_0000/combined_k16.png").is_file()
            )
            self.assertEqual(len(calls), 5)  # Two combined, two FM, one static sheet.
            self.assertTrue(all(call["maximum_mass"] == 1.05 for call in calls))

    def test_only_exact_nfe_path_components_change(self) -> None:
        self.assertEqual(
            plotting._path_at_nfe(
                Path("model_nfe_0256/evaluation/nfe_0256/samples.pt"), 4
            ),
            Path("model_nfe_0256/evaluation/nfe_0004/samples.pt"),
        )
        path = Path("regression/evaluation/samples.pt")
        self.assertEqual(plotting._path_at_nfe(path, 256), path)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            plotting._path_at_nfe(Path("nfe_0001/nfe_0256/samples.pt"), 4)

    def test_skip_missing_never_skips_invalid_artifacts(self) -> None:
        invalid = copy.deepcopy(self.model)
        invalid["specification_hash"] = "wrong"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(self.reference, root / "reference.pt")
            torch.save(invalid, root / "model.pt")
            with self.assertRaisesRegex(ValueError, "specification hash"):
                plotting.main(
                    [
                        "--reference",
                        str(root / "reference.pt"),
                        "--models",
                        f"HFM={root / 'model.pt'}",
                        "--output_dir",
                        str(root / "figures"),
                        "--skip_missing",
                    ]
                )
            self.assertFalse((root / "figures").exists())

    def test_method_colors_do_not_depend_on_comparison_group(self) -> None:
        self.assertEqual(plotting._model_color("HFM"), "#222222")
        self.assertEqual(plotting._model_color("QWFM-L128"), "#1B9E77")
        self.assertEqual(plotting._model_color("QWFM L=512"), "#7570B3")
        self.assertEqual(plotting._model_color("CVAE-z16"), "#E7298A")


if __name__ == "__main__":
    unittest.main()
