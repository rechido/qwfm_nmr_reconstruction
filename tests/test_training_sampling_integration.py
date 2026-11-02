"""Integration checks for restored framework boundaries and strict sample caches."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from codes.data import ContinuousNMRSpec, build_nmr_grid, generate_fixed_split
from codes.evaluate.evaluate_classical_baseline import main as classical_main
from codes.utils.io_utils import load_torch_artifact
from codes.utils.posterior_evaluation_utils import (
    build_sampling_artifact,
    load_and_validate_references,
    load_reusable_sampling_artifact,
    save_or_reuse_sampling_artifact,
)


def _explicit_spec_arguments(spec: ContinuousNMRSpec) -> list[str]:
    return [
        "--num_measurements",
        str(spec.num_measurements),
        "--num_spectrum_bins",
        str(spec.num_spectrum_bins),
        "--rate_minimum",
        str(spec.rate_minimum),
        "--rate_maximum",
        str(spec.rate_maximum),
        "--echo_time_minimum",
        str(spec.echo_time_minimum),
        "--echo_time_maximum",
        str(spec.echo_time_maximum),
        "--noise_standard_deviation",
        str(spec.noise_standard_deviation),
        "--component_count_minimum",
        str(spec.component_count_minimum),
        "--component_count_maximum",
        str(spec.component_count_maximum),
        "--center_minimum",
        str(spec.center_minimum),
        "--center_maximum",
        str(spec.center_maximum),
        "--coefficient_of_variation_minimum",
        str(spec.coefficient_of_variation_minimum),
        "--coefficient_of_variation_maximum",
        str(spec.coefficient_of_variation_maximum),
        "--mixture_weight_concentration",
        str(spec.mixture_weight_concentration),
        "--minimum_component_retained_mass",
        str(spec.minimum_component_retained_mass),
        "--maximum_prior_attempts",
        str(spec.maximum_prior_attempts),
    ]


def _reference_pair(spec: ContinuousNMRSpec) -> tuple[dict, dict]:
    grid = build_nmr_grid(spec)
    generator = torch.Generator().manual_seed(1701)
    groups = 2
    posterior_count = 8
    truth = torch.rand(groups, spec.num_spectrum_bins, generator=generator)
    truth /= truth.sum(dim=-1, keepdim=True)
    measurements = truth @ grid.kernel.to(torch.float32).T
    primary_samples = torch.rand(
        groups, posterior_count, spec.num_spectrum_bins, generator=generator
    )
    primary_samples /= primary_samples.sum(dim=-1, keepdim=True)
    oracle_samples = torch.rand(
        groups, posterior_count, spec.num_spectrum_bins, generator=generator
    )
    oracle_samples /= oracle_samples.sum(dim=-1, keepdim=True)
    common = {
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "condition_ids": torch.tensor([501, 502], dtype=torch.int64),
        "measurements": measurements,
        "truth_histograms": truth,
        "grid": grid.as_artifact(),
        "smc_settings": {"test": True},
    }
    primary = {
        **common,
        "reference_run_id": 0,
        "reference_master_seed": 7001,
        "reference_histograms": primary_samples,
    }
    oracle = {
        **common,
        "reference_run_id": 1,
        "reference_master_seed": 7002,
        "reference_histograms": oracle_samples,
    }
    return primary, oracle


class ReferenceGridValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ContinuousNMRSpec(num_spectrum_bins=8)
        self.primary, self.oracle = _reference_pair(self.spec)

    def _load(self, primary: dict, oracle: dict) -> tuple[dict, dict]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            torch.save(primary, root / "primary.pt")
            torch.save(oracle, root / "oracle.pt")
            return load_and_validate_references(
                root / "primary.pt", root / "oracle.pt", self.spec
            )

    def test_exact_grid_passes(self) -> None:
        self._load(self.primary, self.oracle)

    def test_m64_reference_is_accepted_but_m32_is_rejected(self) -> None:
        self.spec = ContinuousNMRSpec(num_measurements=64, num_spectrum_bins=8)
        primary, oracle = _reference_pair(self.spec)
        loaded, _ = self._load(primary, oracle)
        self.assertEqual(tuple(loaded["measurements"].shape), (2, 64))
        with self.assertRaisesRegex(ValueError, "specification hash"):
            self._load(self.primary, self.oracle)

    def test_grid_roundoff_passes_without_replacing_saved_values(self) -> None:
        for reference_index in (0, 1):
            with self.subTest(reference=reference_index):
                references = [copy.deepcopy(self.primary), copy.deepcopy(self.oracle)]
                grid = references[reference_index]["grid"]
                for key, value in grid.items():
                    if torch.is_tensor(value) and value.is_floating_point():
                        grid[key] = torch.nextafter(
                            value, torch.full_like(value, float("inf"))
                        )
                loaded = self._load(*references)
                for index, reference in enumerate(references):
                    for key, value in reference["grid"].items():
                        if torch.is_tensor(value):
                            self.assertTrue(
                                torch.equal(loaded[index]["grid"][key], value)
                            )

    def test_real_grid_changes_report_field_and_difference(self) -> None:
        for reference_index, name in enumerate(("Primary", "Oracle")):
            for key, value in self.primary["grid"].items():
                if not torch.is_tensor(value) or not value.is_floating_point():
                    continue
                with self.subTest(reference=name, field=key):
                    references = [
                        copy.deepcopy(self.primary),
                        copy.deepcopy(self.oracle),
                    ]
                    references[reference_index]["grid"][key].reshape(-1)[0] += 1.0e-8
                    with self.assertRaisesRegex(
                        ValueError, rf"{name}.*grid\['{key}'\].*max_abs_difference"
                    ):
                        self._load(*references)

    def test_grid_schema_and_nonfinite_values_are_rejected(self) -> None:
        grid = self.primary["grid"]
        cases = [
            ("kernel", grid["kernel"].float(), "dtype"),
            ("kernel", grid["kernel"][:1], "shape"),
            ("kernel", grid["kernel"].tolist(), "must be a tensor"),
            ("kernel", torch.full_like(grid["kernel"], float("nan")), "non-finite"),
            ("kernel", torch.full_like(grid["kernel"], float("inf")), "non-finite"),
            ("echo_subset_indices", grid["echo_subset_indices"] + 1, "exactly"),
            ("echo_subset_indices", grid["echo_subset_indices"].float(), "dtype"),
            ("specification_hash", "wrong-hash", "does not match"),
        ]
        for key, value, message in cases:
            with self.subTest(field=key, message=message):
                primary = copy.deepcopy(self.primary)
                primary["grid"][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    self._load(primary, self.oracle)
        for field_change in ("missing", "extra", "not_mapping"):
            with self.subTest(fields=field_change):
                primary = copy.deepcopy(self.primary)
                if field_change == "missing":
                    del primary["grid"]["kernel"]
                elif field_change == "extra":
                    primary["grid"]["extra"] = 1
                else:
                    primary["grid"] = None
                with self.assertRaisesRegex(ValueError, "grid fields"):
                    self._load(primary, self.oracle)

    def test_condition_identity_remains_exact(self) -> None:
        for key in ("condition_ids", "measurements", "truth_histograms"):
            with self.subTest(field=key):
                oracle = copy.deepcopy(self.oracle)
                value = oracle[key]
                if value.is_floating_point():
                    oracle[key] = torch.nextafter(
                        value, torch.full_like(value, float("inf"))
                    )
                else:
                    oracle[key] = value + 1
                with self.assertRaisesRegex(ValueError, f"differ in {key}"):
                    self._load(self.primary, oracle)


class SamplingArtifactTests(unittest.TestCase):
    def test_strict_cache_identity_and_condition_major_shape(self) -> None:
        spec = ContinuousNMRSpec(num_spectrum_bins=8)
        primary, oracle = _reference_pair(spec)
        samples = primary["reference_histograms"][:, :4].clone()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "checkpoints" / "checkpoint-000000001"
            artifact = build_sampling_artifact(
                spec=spec,
                primary_reference=primary,
                samples=samples,
                method="cache_test",
                training_config={"experiment": "cache_test"},
                output_dir=root,
                checkpoint=checkpoint,
                model_subfolder="unet",
                primary_reference_path=root / "primary.pt",
                state_size=8,
                deterministic=False,
                sampling_config={"sampler": "test"},
                source_seed=17,
                sampling_wall_seconds=0.5,
                inference_settings={"world_size": 1},
                sampler_diagnostic_totals={"draw_count": 8.0},
            )
            destination = root / "evaluation" / "samples.pt"
            saved = save_or_reuse_sampling_artifact(
                destination,
                artifact,
                reuse_existing=False,
                primary_reference=primary,
                oracle_reference=oracle,
                sample_count=4,
            )
            self.assertEqual(tuple(saved["samples"].shape), (2, 4, 8))

            dynamic = {
                "samples",
                "sampling_wall_seconds",
                "sampler_diagnostic_totals",
                "generated_unique_count",
                "maximum_mass_absolute_error",
                "representation_metadata",
            }
            expected = {
                key: value for key, value in artifact.items() if key not in dynamic
            }
            reused = load_reusable_sampling_artifact(
                destination,
                expected_provenance=expected,
                primary_reference=primary,
                oracle_reference=oracle,
                sample_count=4,
            )
            torch.testing.assert_close(reused["samples"], samples)

            stale = dict(expected)
            stale["source_seed"] = 18
            with self.assertRaisesRegex(ValueError, "source_seed"):
                load_reusable_sampling_artifact(
                    destination,
                    expected_provenance=stale,
                    primary_reference=primary,
                    oracle_reference=oracle,
                    sample_count=4,
                )

            # Grid tolerance must not weaken identity of the saved sample cache.
            stale_grid = copy.deepcopy(expected)
            kernel = stale_grid["grid"]["kernel"]
            stale_grid["grid"]["kernel"] = torch.nextafter(
                kernel, torch.full_like(kernel, float("inf"))
            )
            with self.assertRaisesRegex(ValueError, "stale provenance field 'grid'"):
                load_reusable_sampling_artifact(
                    destination,
                    expected_provenance=stale_grid,
                    primary_reference=primary,
                    oracle_reference=oracle,
                    sample_count=4,
                )


class ClassicalCacheTests(unittest.TestCase):
    def test_classical_reuse_validates_solver_k_schema_and_reference(self) -> None:
        spec = ContinuousNMRSpec(num_spectrum_bins=8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = generate_fixed_split(
                num_samples=2,
                spec=spec,
                master_seed=717,
                split="classical_cache_test",
            )
            reference["condition_ids"] = torch.tensor([31, 32], dtype=torch.int64)
            reference["measurements"] = reference.pop("measurement")
            reference_path = root / "reference.pt"
            torch.save(reference, reference_path)
            output = root / "output"
            common = [
                "--reference_path",
                str(reference_path),
                "--output_dir",
                str(output),
                "--method",
                "nnls",
                *_explicit_spec_arguments(spec),
                "--num_samples_per_condition",
                "3",
                "--num_workers",
                "1",
                "--reuse_existing",
            ]
            classical_main(common)
            original = (output / "samples.pt").read_bytes()
            classical_main(common)
            self.assertEqual((output / "samples.pt").read_bytes(), original)

            stale_k = list(common)
            k_index = stale_k.index("--num_samples_per_condition") + 1
            stale_k[k_index] = "4"
            with self.assertRaisesRegex(ValueError, "num_samples_per_condition"):
                classical_main(stale_k)

            stale_options = [*common, "--x2_factor", "1.03"]
            with self.assertRaisesRegex(ValueError, "sampling_config"):
                classical_main(stale_options)

            artifact = load_torch_artifact(output / "samples.pt")
            artifact["artifact_schema_version"] = -1
            torch.save(artifact, output / "samples.pt")
            with self.assertRaisesRegex(ValueError, "artifact_schema_version"):
                classical_main(common)


if __name__ == "__main__":
    unittest.main()
