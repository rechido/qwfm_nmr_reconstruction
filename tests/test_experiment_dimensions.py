"""Dimension contracts shared by model discovery and both evaluation paths."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch

from codes.data import ContinuousNMRSpec, generate_fixed_split
from codes.evaluate.audit_reference import validate_reference_artifacts
from codes.evaluate.evaluate_saved_samples import validate_evaluation_artifacts
from codes.evaluate.visualize_posterior_samples import validate_visualization_artifacts
from codes.metrics import compute_energy_score_report, compute_kernel_score_report, compute_sample_set_metrics
from codes.utils.model_variant_utils import VARIANTS, experiment_dimensions, resolve_training_dir


EXPERIMENT_DIMENSIONS = ((32, 128), (64, 128), (64, 256), (128, 256))


def _artifacts(spec: ContinuousNMRSpec) -> tuple[dict, dict, dict, dict]:
    truth = generate_fixed_split(num_samples=2, spec=spec, master_seed=91, split="posterior_test")
    common = {
        "protocol_id": spec.protocol_id,
        "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash,
        "condition_ids": truth["sample_id"],
        "measurements": truth["measurement"],
        "grid": truth["grid"],
    }
    generator = torch.Generator().manual_seed(19)

    def cloud() -> torch.Tensor:
        values = torch.rand(2, 16, spec.num_spectrum_bins, generator=generator)
        return values / values.sum(-1, keepdim=True)

    references = [
        {
            **common, "reference_run_id": run_id, "reference_master_seed": 101,
            "smc_settings": {"fixture": True},
            "truth_histograms": truth["histogram"],
            "truth_component_counts": truth["component_count"],
            "reference_histograms": cloud(),
            "posterior_component_count_probability": torch.full((2, 5), 0.2),
        }
        for run_id in (0, 1)
    ]
    samples = {**common, "samples": cloud(), "num_samples_per_condition": 16}
    return samples, references[0], references[1], truth


class ExperimentDimensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_all_variants_resolve_model_names_for_all_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for measurements, bins in EXPERIMENT_DIMENSIONS:
                experiment = Path(directory) / f"m{measurements}n{bins}"
                self.assertEqual(experiment_dimensions(experiment), (measurements, bins))
                for variant, (_, template, _, _) in VARIANTS.items():
                    with self.subTest(experiment=experiment.name, variant=variant):
                        resolved = resolve_training_dir(experiment, variant, require_config=False)
                        self.assertEqual(resolved.parent, experiment)
                        self.assertEqual(resolved.name, template.format(shape=f"m{measurements}_n{bins}"))
            for name in ("custom", "m128n256_extra", "m16n128", "m128n255"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    experiment_dimensions(Path(directory) / name)

    def test_both_reference_consumers_and_visualization_accept_all_grids(self) -> None:
        for measurements, bins in EXPERIMENT_DIMENSIONS:
            with self.subTest(measurements=measurements, bins=bins):
                spec = ContinuousNMRSpec(num_measurements=measurements, num_spectrum_bins=bins)
                samples, primary, oracle, _ = _artifacts(spec)
                self.assertEqual(validate_reference_artifacts([primary, oracle], sample_count=8), spec)
                self.assertEqual(validate_evaluation_artifacts(samples, primary, oracle, sample_count=8), spec)
                self.assertEqual(validate_visualization_artifacts(primary, {"model": samples}), spec)
                key = f"echo_times_{spec.canonical_echo_count}"
                broken_primary = copy.deepcopy(primary)
                broken_primary["grid"][key][0] += 0.1
                with self.assertRaisesRegex(ValueError, key):
                    validate_reference_artifacts([broken_primary, oracle], sample_count=8)
                with self.assertRaisesRegex(ValueError, key):
                    validate_evaluation_artifacts(samples, broken_primary, oracle, sample_count=8)
                with self.assertRaisesRegex(ValueError, key):
                    validate_visualization_artifacts(broken_primary, {"model": samples})

    def test_256_bin_distribution_and_truth_scores_are_finite(self) -> None:
        for measurements in (64, 128):
            spec = ContinuousNMRSpec(num_measurements=measurements, num_spectrum_bins=256)
            samples, primary, _, truth = _artifacts(spec)
            for metric in ("l2", "w1", "w2"):
                with self.subTest(measurements=measurements, metric=metric):
                    scores = compute_sample_set_metrics(
                        primary["reference_histograms"][0], samples["samples"][0],
                        metric=metric, log_rate_edges=truth["grid"]["log_rate_edges"],
                        num_quantiles=256, chunk_size=8,
                    )
                    self.assertTrue(all(math.isfinite(value) for value in scores.values()))
                    self.assertTrue({"precision", "recall", "density", "coverage",
                                     "validity_distance", "coverage_distance"}.issubset(scores))
            for score_function in (compute_energy_score_report, compute_kernel_score_report):
                options = {"bandwidth": 0.2} if score_function is compute_kernel_score_report else {}
                report = score_function(samples, truth, sample_count=8, **options)
                self.assertTrue(report["available"])
                self.assertEqual(report["condition_count"], 2)
                self.assertEqual(report["specification_hash"], spec.specification_hash)


if __name__ == "__main__":
    unittest.main()
