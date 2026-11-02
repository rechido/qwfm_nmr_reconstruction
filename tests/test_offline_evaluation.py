import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.evaluate.evaluate_saved_samples import (
    EvaluationSettings,
    evaluate_saved_samples_artifacts,
    main as evaluation_main,
    validate_evaluation_artifacts,
)
from codes.evaluate.artifact_contract import classical_representation_metadata
from codes.evaluate.evaluate_classical_baseline import main as classical_main
import codes.evaluate.visualize_posterior_samples as visualization_module
from codes.evaluate.visualize_posterior_samples import visualize_posterior_artifacts


class OfflineArtifactTests(unittest.TestCase):
    def assert_nested_close(self, first, second) -> None:
        if isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_close(first[key], second[key])
        elif isinstance(first, list):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_close(left, right)
        elif isinstance(first, float):
            self.assertTrue(
                math.isclose(first, second, rel_tol=1.0e-12, abs_tol=1.0e-12),
                msg=f"{first!r} != {second!r}",
            )
        else:
            self.assertEqual(first, second)

    def setUp(self) -> None:
        self.spec = ContinuousNMRSpec(num_spectrum_bins=8)
        self.grid = build_nmr_grid(self.spec)
        generator = torch.Generator().manual_seed(7701)
        groups = 2
        reference_count = 24
        generated_count = 16

        truth = torch.rand(
            (groups, self.spec.num_spectrum_bins),
            generator=generator,
            dtype=torch.float64,
        )
        truth /= truth.sum(dim=-1, keepdim=True)
        measurements = (truth @ self.grid.kernel.T).to(torch.float32)
        primary_samples = torch.rand(
            (groups, reference_count, self.spec.num_spectrum_bins),
            generator=generator,
            dtype=torch.float64,
        )
        primary_samples /= primary_samples.sum(dim=-1, keepdim=True)
        oracle_samples = torch.rand(
            (groups, reference_count, self.spec.num_spectrum_bins),
            generator=generator,
            dtype=torch.float64,
        )
        oracle_samples /= oracle_samples.sum(dim=-1, keepdim=True)
        model_samples = torch.rand(
            (groups, generated_count, self.spec.num_spectrum_bins),
            generator=generator,
            dtype=torch.float64,
        )
        model_samples /= model_samples.sum(dim=-1, keepdim=True)
        condition_ids = torch.tensor([701, 702], dtype=torch.int64)

        common_reference = {
            "protocol_id": self.spec.protocol_id,
            "specification": self.spec.to_dict(),
            "specification_hash": self.spec.specification_hash,
            "condition_ids": condition_ids,
            "measurements": measurements,
            "truth_histograms": truth,
            "grid": self.grid.as_artifact(),
        }
        self.primary = {
            **common_reference,
            "reference_run_id": 0,
            "reference_master_seed": 9001,
            "smc_settings": {"tiny": True},
            "reference_histograms": primary_samples,
        }
        self.oracle = {
            **common_reference,
            "reference_run_id": 1,
            "reference_master_seed": 9001,
            "smc_settings": {"tiny": True},
            "reference_histograms": oracle_samples,
        }
        self.samples = {
            "protocol_id": self.spec.protocol_id,
            "specification": self.spec.to_dict(),
            "specification_hash": self.spec.specification_hash,
            "condition_ids": condition_ids.clone(),
            "measurements": measurements.clone(),
            "samples": model_samples,
            "grid": self.grid.as_artifact(),
            "method": "tiny_model",
        }

    def test_repeated_evaluation_is_condition_wise_and_reproducible(self) -> None:
        settings = EvaluationSettings(
            sample_count=4,
            reference_repeats=2,
            nearest_k=1,
            w2_num_quantiles=16,
            distance_chunk_size=2,
            seed=99,
        )
        first_raw, first_summary = evaluate_saved_samples_artifacts(
            self.samples,
            self.primary,
            self.oracle,
            settings=settings,
        )
        second_raw, second_summary = evaluate_saved_samples_artifacts(
            self.samples,
            self.primary,
            self.oracle,
            settings=settings,
        )
        self.assert_nested_close(first_raw, second_raw)
        self.assert_nested_close(first_summary, second_summary)
        self.assertEqual(len(first_raw["records"]), 4)
        self.assertEqual(
            {
                (record["condition_index"], record["repeat_index"])
                for record in first_raw["records"]
            },
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        first_record = first_raw["records"][0]
        self.assertEqual(
            set(first_record["model"]["ground_metrics"]), {"l2", "w1", "w2"}
        )
        self.assertIn("model_minus_oracle", first_record)
        self.assertIn("rate_2_10", first_record["model"]["derived_metrics"])
        self.assertEqual(first_summary["record_count"], 4)

    def test_artifact_validation_catches_identity_failures(self) -> None:
        with self.subTest("same reference run"):
            same_run = dict(self.oracle)
            same_run["reference_run_id"] = 0
            with self.assertRaisesRegex(ValueError, "distinct run IDs"):
                validate_evaluation_artifacts(
                    self.samples, self.primary, same_run, sample_count=4
                )
        with self.subTest("specification hash"):
            wrong_hash = dict(self.samples)
            wrong_hash["specification_hash"] = "not-the-reference-hash"
            with self.assertRaisesRegex(ValueError, "specification hash"):
                validate_evaluation_artifacts(
                    wrong_hash, self.primary, self.oracle, sample_count=4
                )
        with self.subTest("measurement"):
            wrong_measurement = dict(self.samples)
            wrong_measurement["measurements"] = self.samples["measurements"].clone()
            wrong_measurement["measurements"][0, 0] += 0.01
            with self.assertRaisesRegex(ValueError, "measurements differ"):
                validate_evaluation_artifacts(
                    wrong_measurement, self.primary, self.oracle, sample_count=4
                )
        with self.subTest("grid"):
            wrong_grid = copy.deepcopy(self.samples)
            wrong_grid["grid"]["kernel"][0, 0] += 0.01
            with self.assertRaisesRegex(ValueError, r"grid\['kernel'\]"):
                validate_evaluation_artifacts(
                    wrong_grid, self.primary, self.oracle, sample_count=4
                )

    def test_raw_forward_representation_is_strict_and_used_only_for_residual(self) -> None:
        raw_forward = 2.0 * self.samples["samples"][:, :1, :]
        with_raw = copy.deepcopy(self.samples)
        with_raw["forward_spectra"] = raw_forward
        with self.subTest("metadata is mandatory"):
            with self.assertRaisesRegex(ValueError, "representation_metadata"):
                validate_evaluation_artifacts(
                    with_raw, self.primary, self.oracle, sample_count=4
                )
        with_raw["representation_metadata"] = classical_representation_metadata()
        settings = EvaluationSettings(
            sample_count=4,
            reference_repeats=1,
            nearest_k=1,
            w2_num_quantiles=8,
            distance_chunk_size=2,
        )
        raw_result, _ = evaluate_saved_samples_artifacts(
            with_raw, self.primary, self.oracle, settings=settings
        )
        normalized_result, _ = evaluate_saved_samples_artifacts(
            self.samples, self.primary, self.oracle, settings=settings
        )
        self.assertEqual(
            raw_result["records"][0]["model"]["ground_metrics"],
            normalized_result["records"][0]["model"]["ground_metrics"],
        )
        self.assertNotEqual(
            raw_result["records"][0]["model"]["forward_residual"],
            normalized_result["records"][0]["model"]["forward_residual"],
        )
        self.assertEqual(
            raw_result["representation_usage"]["model_forward_residual"]
            ["normalization"],
            "none",
        )

        for bad_value, message in (
            (torch.full_like(raw_forward, float("nan")), "non-finite"),
            (-raw_forward, "negative"),
            (raw_forward[:, :, :-1], "shape"),
        ):
            malformed = copy.deepcopy(with_raw)
            malformed["forward_spectra"] = bad_value
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_evaluation_artifacts(
                        malformed, self.primary, self.oracle, sample_count=4
                    )

    def test_visualizer_uses_raw_forward_but_normalized_spectrum_samples(self) -> None:
        artifact = copy.deepcopy(self.samples)
        artifact["forward_spectra"] = 3.0 * artifact["samples"][:, :1, :]
        artifact["representation_metadata"] = classical_representation_metadata()
        captured: dict[str, torch.Tensor] = {}

        def capture_forward(reference, models, *args, path, **kwargs):
            del reference, args, kwargs
            captured.update(models)
            Path(path).touch()

        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(
                visualization_module,
                "_plot_forward_residuals",
                side_effect=capture_forward,
            ):
                manifest = visualize_posterior_artifacts(
                    self.primary,
                    {"Raw classical": artifact},
                    temporary_directory,
                    condition_indices=[0],
                    dpi=72,
                    include_diagnostics=True,
                )
        torch.testing.assert_close(captured["Raw classical"], artifact["forward_spectra"][0])
        self.assertEqual(
            manifest["representation_usage"]["forward_residual_by_model"]
            ["Raw classical"]["normalization"],
            "none",
        )
        self.assertTrue(manifest["include_diagnostics"])
        self.assertEqual(
            set(manifest["conditions"][0]["diagnostics"]),
            {"combined_posterior_band", "forward_residual", "downstream_quantities"},
        )
        for path in manifest["conditions"][0]["diagnostics"].values():
            self.assertEqual(Path(path).parent.name, "diagnostics")

    def test_classical_cli_serializes_raw_forward_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.pt"
            torch.save(self.primary, reference_path)
            output = root / "classical"
            classical_main(
                [
                    "--reference_path",
                    str(reference_path),
                    "--output_dir",
                    str(output),
                    "--method",
                    "nnls",
                    "--num_measurements",
                    str(self.spec.num_measurements),
                    "--num_spectrum_bins",
                    str(self.spec.num_spectrum_bins),
                    "--rate_minimum",
                    str(self.spec.rate_minimum),
                    "--rate_maximum",
                    str(self.spec.rate_maximum),
                    "--echo_time_minimum",
                    str(self.spec.echo_time_minimum),
                    "--echo_time_maximum",
                    str(self.spec.echo_time_maximum),
                    "--noise_standard_deviation",
                    str(self.spec.noise_standard_deviation),
                    "--component_count_minimum",
                    str(self.spec.component_count_minimum),
                    "--component_count_maximum",
                    str(self.spec.component_count_maximum),
                    "--center_minimum",
                    str(self.spec.center_minimum),
                    "--center_maximum",
                    str(self.spec.center_maximum),
                    "--coefficient_of_variation_minimum",
                    str(self.spec.coefficient_of_variation_minimum),
                    "--coefficient_of_variation_maximum",
                    str(self.spec.coefficient_of_variation_maximum),
                    "--mixture_weight_concentration",
                    str(self.spec.mixture_weight_concentration),
                    "--minimum_component_retained_mass",
                    str(self.spec.minimum_component_retained_mass),
                    "--maximum_prior_attempts",
                    str(self.spec.maximum_prior_attempts),
                    "--num_samples_per_condition",
                    "4",
                ]
            )
            artifact = torch.load(output / "samples.pt", weights_only=False)
            self.assertEqual(tuple(artifact["forward_spectra"].shape), (2, 1, 8))
            torch.testing.assert_close(
                artifact["forward_spectra"][:, 0],
                torch.from_numpy(artifact["classical_results"]["raw_spectra"]).float(),
            )
            torch.testing.assert_close(
                artifact["samples"][:, 0],
                torch.from_numpy(
                    artifact["classical_results"]["normalized_spectra"]
                ).float(),
            )

    def test_cli_saves_raw_and_aggregate_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_path = root / "samples.pt"
            primary_path = root / "primary.pt"
            oracle_path = root / "oracle.pt"
            torch.save(self.samples, sample_path)
            torch.save(self.primary, primary_path)
            torch.save(self.oracle, oracle_path)
            output = root / "metrics"
            evaluation_main(
                [
                    "--samples",
                    str(sample_path),
                    "--primary_reference",
                    str(primary_path),
                    "--oracle_reference",
                    str(oracle_path),
                    "--output_dir",
                    str(output),
                    "--sample_count",
                    "4",
                    "--reference_repeats",
                    "1",
                    "--nearest_k",
                    "1",
                    "--w2_num_quantiles",
                    "8",
                    "--distance_chunk_size",
                    "2",
                ]
            )
            for name in ("evaluation_raw.json", "evaluation_summary.json"):
                path = output / name
                self.assertTrue(path.is_file())
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.assertEqual(
                    payload["specification_hash"], self.spec.specification_hash
                )

    def test_visualizer_creates_only_comparison_units(self) -> None:
        second_model = copy.deepcopy(self.samples)
        second_model["samples"] = torch.flip(second_model["samples"], dims=(-1,))
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = visualize_posterior_artifacts(
                self.primary,
                {"Model A": self.samples, "Model B": second_model},
                temporary_directory,
                condition_indices=[0],
                dpi=72,
            )
            condition_dir = Path(temporary_directory) / "condition_0000"
            expected = {
                "combined_k16.png",
                "Model_A_k16.png",
                "Model_B_k16.png",
            }
            self.assertEqual(
                {path.name for path in condition_dir.glob("*.png")}, expected
            )
            self.assertEqual(len(manifest["conditions"]), 1)
            self.assertEqual(manifest["conditions"][0]["condition_index"], 0)
            self.assertEqual(manifest["conditions"][0]["condition_id"], 701)
            self.assertEqual(manifest["qualitative_sample_count"], 16)
            self.assertFalse(manifest["include_diagnostics"])
            self.assertEqual(manifest["conditions"][0]["diagnostics"], {})
            self.assertEqual(
                manifest["conditions"][0]["displayed_shots_by_model"],
                {"Model A": 16, "Model B": 16},
            )
            self.assertEqual(
                manifest["conditions"][0]["reference_sample_indices"],
                list(range(16)),
            )
            self.assertTrue(
                (Path(temporary_directory) / "visualization_manifest.json").is_file()
            )

    def test_kshot_gallery_retains_condition_truth_and_every_sample(self) -> None:
        reference_samples = self.primary["reference_histograms"][0, :16]
        first_model = self.samples["samples"][0]
        second_model = torch.flip(first_model, dims=(-1,))
        truth = self.primary["truth_histograms"][0]
        measurement = self.primary["measurements"][0]
        captured = []

        def capture_figure(figure, path, *, dpi):
            del path, dpi
            for axis in figure.axes:
                self.assertEqual(len(axis.lines), 1)
                line = axis.lines[0]
                captured.append(
                    (
                        torch.as_tensor(line.get_xdata()).clone(),
                        torch.as_tensor(line.get_ydata()).clone(),
                    )
                )

        with mock.patch.object(
            visualization_module, "_save_figure", side_effect=capture_figure
        ):
            visualization_module._plot_kshot_gallery(
                reference_samples,
                {"Model A": first_model, "Model B": second_model},
                truth,
                measurement,
                self.grid.rate_centers,
                self.grid.echo_times,
                condition_id=701,
                path=Path("unused.png"),
                maximum_mass=1.0,
                dpi=72,
                combined=True,
                nfe=16,
            )

        self.assertEqual(len(captured), 2 + 16 + 16 + 16)
        torch.testing.assert_close(captured[0][0], self.grid.echo_times * 1000)
        torch.testing.assert_close(captured[0][1], measurement)
        expected_spectra = torch.cat(
            (truth[None], reference_samples, first_model, second_model), dim=0
        )
        for (rates, spectrum), expected in zip(captured[1:], expected_spectra):
            torch.testing.assert_close(rates, self.grid.rate_centers)
            torch.testing.assert_close(spectrum, expected)

    def test_visualizer_rejects_filename_collisions_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "collide"):
                visualize_posterior_artifacts(
                    self.primary,
                    {"Model A": self.samples, "Model_A": self.samples},
                    temporary_directory,
                    condition_indices=[0],
                    dpi=72,
                )
            condition_dir = Path(temporary_directory) / "condition_0000"
            condition_dir.mkdir(parents=True, exist_ok=True)
            (condition_dir / "combined_k16.png").touch()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                visualize_posterior_artifacts(
                    self.primary,
                    {"Model A": self.samples},
                    temporary_directory,
                    condition_indices=[0],
                    dpi=72,
                )


if __name__ == "__main__":
    unittest.main()
