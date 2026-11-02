"""Small saved-artifact fixtures exercise paper figures without model execution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import find_peaks
import torch

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.evaluate import plot_transport_paths as paths
from codes.evaluate import visualize_paper_reconstructions as figures
from codes.utils.nmr_utils import (
    decode_quantiles, histogram_to_quantiles, make_normalized_log_rate_edges,
)
from codes.utils.predictive_generation_utils import (
    PROPER_SCORE_PROTOCOL_VERSION, PROPER_SCORE_TEST_SPLIT, _save_cache, file_sha256,
)
from codes.utils.quantile_decoder_artifacts import decoder_comparison_metadata


def saved_fixture(experiment: Path) -> tuple[dict, str]:
    """Five hand-built morphologies, not a generated benchmark dataset."""
    spec = ContinuousNMRSpec(num_measurements=64, num_spectrum_bins=128)
    grid = build_nmr_grid(spec).as_artifact()
    edges = grid["log_rate_edges"].double()
    centers = .5 * (edges[1:] + edges[:-1])
    histograms = []
    for count in range(1, 6):
        locations = torch.tensor([centers.mean()]) if count == 1 else torch.linspace(
            edges[0] + .55, edges[-1] - .55, count)
        width = .30 if count == 1 else .065
        histogram = torch.exp(-.5 * ((centers[:, None] - locations[None, :]) / width) ** 2).sum(1)
        histograms.append((histogram / histogram.sum()).float())
    histogram = torch.stack(histograms)
    clean = histogram.double() @ grid["kernel"].double().T
    noise = .001 * torch.sin(torch.arange(5 * 64).reshape(5, 64).double())
    active = torch.arange(5)[None, :] < torch.arange(1, 6)[:, None]
    weights = active.float() / torch.arange(1, 6)[:, None]
    truth = {
        "protocol_id": spec.protocol_id, "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash, "grid": grid,
        "sample_id": torch.arange(5), "histogram": histogram,
        "measurement": (clean + noise).float(), "clean_measurement": clean.float(),
        "noise": noise.float(), "component_count": torch.arange(1, 6),
        "active_mask": active, "mixture_weights": weights,
        "master_seed": 123, "split": PROPER_SCORE_TEST_SPLIT,
        "proper_score_provenance": {
            "protocol_version": PROPER_SCORE_PROTOCOL_VERSION,
            "num_conditions": 5, "master_seed": 123,
            "split": PROPER_SCORE_TEST_SPLIT, "specification_hash": spec.specification_hash,
        },
    }
    truth_path = experiment / "proper_scores/g5_k8/test_conditions.pt"
    _save_cache(truth_path, truth)
    truth_hash = file_sha256(truth_path)
    for variant, label, _ in figures.MODELS:
        method = figures.VARIANTS[variant][0]
        deterministic = label == "Regression"
        count = 1 if deterministic else 8
        samples = torch.stack([histogram.roll(draw - 3, dims=-1) for draw in range(count)], dim=1)
        nfe = 256 if method in figures.FLOW_METHODS else None
        provenance = {
            "protocol_version": PROPER_SCORE_PROTOCOL_VERSION, "method": method,
            "variant": variant, "conditions_sha256": truth_hash,
            "requested_sample_count": 8, "stored_sample_count": count, "nfe": nfe,
        }
        artifact = {
            "protocol_id": spec.protocol_id, "specification": spec.to_dict(),
            "specification_hash": spec.specification_hash, "grid": grid,
            "method": method, "deterministic": deterministic,
            "condition_ids": truth["sample_id"], "measurements": truth["measurement"],
            "samples": samples, "sampling_config": {"ode_steps": nfe},
            "proper_score_provenance": provenance,
        }
        if label == "QWFM":
            normalized = make_normalized_log_rate_edges(grid["log_rate_edges"])
            quantiles = histogram_to_quantiles(samples.flatten(0, 1), normalized, num_quantiles=128).reshape(5, 8, 128)
            samples = decode_quantiles(quantiles.flatten(0, 1), normalized, decoder="linear_cdf").reshape(5, 8, 128)
            metadata = decoder_comparison_metadata(quantiles, "linear_cdf")
            artifact.update(samples=samples, state_size=128, quantiles=quantiles,
                            decoder_comparison=metadata, quantile_sha256=metadata["quantile_sha256"])
            artifact["sampling_config"]["histogram_decoder"] = metadata["decoder_signature"]
            provenance.update(saved_quantiles=True, decoder_signature=metadata["decoder_signature"],
                              quantile_sha256=metadata["quantile_sha256"])
        target = figures.prediction_path(experiment, "g5_k8", variant, (64, 128), 256)
        _save_cache(target, artifact)
    return truth, truth_hash


class ReconstructionFigureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.experiment = self.directory / "m64n128"
        self.truth, self.truth_hash = saved_fixture(self.experiment)

    def test_truth_only_selection_and_missing_morphology(self):
        with patch.object(torch, "load", side_effect=AssertionError("Selection must not open predictions")):
            selected = figures.select_cases(self.truth)
        self.assertEqual([row["component_count"] for row in selected], [1, 2, 3, 4, 5])
        self.assertEqual([row["condition_id"] for row in selected], [0, 1, 2, 3, 4])
        invalid = dict(self.truth, component_count=torch.ones(5, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "Two separated peaks"):
            figures.select_cases(invalid)

    def test_routes_linear_decoder_and_reuses_original_noise(self):
        selected = figures.select_cases(self.truth)
        arrays, sources = figures.collect_display_arrays(
            self.truth, self.truth_hash, selected, self.experiment, 256, 8)
        self.assertIn("decoder_comparison/linear_cdf", sources["QWFM"]["samples_path"])
        self.assertFalse((self.experiment / "proper_scores/g5_k8" /
                          "quantile_wasserstein_m64_n128_l128_unet64x4_t1024").exists())
        for _, label, _ in figures.MODELS:
            signal = arrays[f"{label}_forward_signal"]
            expected = signal + arrays["original_noise"][:, None, :]
            np.testing.assert_array_equal(arrays[f"{label}_shared_noise_signal"], expected)
            np.testing.assert_allclose(expected - arrays["observation"][:, None, :],
                                       signal - arrays["clean_signal"][:, None, :], atol=2e-16, rtol=0)
            if label != "QWFM":
                self.assertNotIn("decoder_comparison", sources[label]["samples_path"])
        self.assertEqual(arrays["QWFM_histogram"].shape, (5, 8, 128))
        self.assertEqual(arrays["Regression_histogram"].shape, (5, 1, 128))

    def test_wrong_decoding_is_rejected_even_with_a_valid_file_checksum(self):
        path = figures.prediction_path(self.experiment, "g5_k8", "quantile_wasserstein_l128", (64, 128), 256)
        artifact = torch.load(path, weights_only=False)
        artifact["samples"] = artifact["samples"].roll(1, dims=-1)
        _save_cache(path, artifact)
        with self.assertRaisesRegex(ValueError, "do not match"):
            figures.collect_display_arrays(self.truth, self.truth_hash, figures.select_cases(self.truth),
                                           self.experiment, 256, 8)

    def test_render_five_pages_without_input_changes_or_network(self):
        inputs = {path: file_sha256(path) for path in self.experiment.rglob("*") if path.is_file()}
        output = self.directory / "figures"
        with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("No network")):
            record = figures.main([
                "--experiment_dir", str(self.experiment), "--output_dir", str(output),
                "--num_conditions", "5", "--sample_count", "8", "--nfe", "256",
                "--num_cpu_threads", "1",
            ])
        self.assertEqual(len(list(output.glob("*.png"))), 5)
        self.assertEqual(len(list(output.glob("*.pdf"))), 5)
        with (output / "selected_cases.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["condition_id"] for row in rows], ["0", "1", "2", "3", "4"])
        self.assertEqual(inputs, {path: file_sha256(path) for path in inputs})
        self.assertTrue(record["no_model_inference"])
        self.assertTrue(record["sources"]["QWFM"]["displayed_histograms_match_saved_quantiles"])
        self.assertEqual(record, json.loads((output / "qualitative_provenance.json").read_text()))


class TransportFigureTests(unittest.TestCase):
    def test_matched_endpoints_normalized_mass_and_distinct_midpoint_paths(self):
        x, times, histogram, quantile = paths.transport_paths()
        np.testing.assert_allclose(histogram[[0, -1]], quantile[[0, -1]], rtol=0, atol=0)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        for path in (histogram, quantile):
            np.testing.assert_allclose(integrate(path, x, axis=1), 1, atol=5e-16, rtol=0)
        midpoint = len(times) // 2
        self.assertEqual(len(find_peaks(histogram[midpoint], prominence=.1)[0]), 2)
        self.assertEqual(len(find_peaks(quantile[midpoint], prominence=.1)[0]), 1)

    def test_standalone_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = paths.main(["--output_dir", temporary, "--dpi", "80"])
            for path in record["files"].values():
                self.assertGreater(Path(path).stat().st_size, 1000)
            self.assertFalse(record["model_execution"])


if __name__ == "__main__":
    unittest.main()
