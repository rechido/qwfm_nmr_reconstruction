"""Model-local reconstruction plots read only explicitly selected saved samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from codes.data.grid import build_nmr_grid
from codes.data.spec import ContinuousNMRSpec
from codes.evaluate import visualize_paper_reconstructions as legacy
from codes.evaluate import visualize_reconstructions as plots
from codes.utils.nmr_utils import decode_quantiles, histogram_to_quantiles, make_normalized_log_rate_edges
from codes.utils.predictive_generation_utils import (
    PROPER_SCORE_PROTOCOL_VERSION, PROPER_SCORE_TEST_SPLIT, _save_cache, file_sha256,
)
from codes.utils.quantile_decoder_artifacts import decoder_comparison_metadata


def saved_fixture(experiment: Path, dimensions=(64, 128), models=plots.MODEL_NAMES):
    """Five deterministic morphologies with no training/checkpoint/SMC files."""
    m, n = dimensions
    spec = ContinuousNMRSpec(num_measurements=m, num_spectrum_bins=n)
    grid = build_nmr_grid(spec).as_artifact()
    edges = grid["log_rate_edges"].double()
    centers = .5 * (edges[1:] + edges[:-1])
    histograms = []
    for count in range(1, 6):
        locations = torch.tensor([centers.mean()]) if count == 1 else torch.linspace(
            edges[0] + .55, edges[-1] - .55, count)
        width = .30 if count == 1 else .065
        values = torch.exp(-.5 * ((centers[:, None] - locations[None, :]) / width) ** 2).sum(1)
        histograms.append((values / values.sum()).float())
    histogram = torch.stack(histograms)
    clean = histogram.double() @ grid["kernel"].double().T
    noise = .001 * torch.sin(torch.arange(5 * m).reshape(5, m).double())
    active = torch.arange(5)[None, :] < torch.arange(1, 6)[:, None]
    truth = {
        "protocol_id": spec.protocol_id, "specification": spec.to_dict(),
        "specification_hash": spec.specification_hash, "grid": grid,
        "sample_id": torch.arange(5), "histogram": histogram,
        "measurement": (clean + noise).float(), "clean_measurement": clean.float(),
        "noise": noise.float(), "component_count": torch.arange(1, 6), "active_mask": active,
        "mixture_weights": active.float() / torch.arange(1, 6)[:, None],
        "master_seed": 123, "split": PROPER_SCORE_TEST_SPLIT,
        "proper_score_provenance": {"protocol_version": PROPER_SCORE_PROTOCOL_VERSION,
            "num_conditions": 5, "master_seed": 123, "split": PROPER_SCORE_TEST_SPLIT,
            "specification_hash": spec.specification_hash},
    }
    truth_path = experiment / "proper_scores/g5_k8/test_conditions.pt"
    _save_cache(truth_path, truth)
    truth_hash = file_sha256(truth_path)
    for variant, label, _ in legacy.models_for_quantiles(n):
        if label not in models:
            continue
        method = legacy.VARIANTS[variant][0]
        count = 1 if label == "Regression" else 8
        samples = torch.stack([histogram.roll(draw - 3, dims=-1) for draw in range(count)], dim=1)
        nfe = 256 if method in legacy.FLOW_METHODS else None
        provenance = {"protocol_version": PROPER_SCORE_PROTOCOL_VERSION, "method": method,
                      "variant": variant, "conditions_sha256": truth_hash,
                      "requested_sample_count": 8, "stored_sample_count": count, "nfe": nfe}
        artifact = {"protocol_id": spec.protocol_id, "specification": spec.to_dict(),
            "specification_hash": spec.specification_hash, "grid": grid, "method": method,
            "deterministic": label == "Regression", "condition_ids": truth["sample_id"],
            "measurements": truth["measurement"], "samples": samples,
            "sampling_config": {"ode_steps": nfe}, "proper_score_provenance": provenance}
        if label == "QWFM":
            normalized = make_normalized_log_rate_edges(grid["log_rate_edges"])
            quantiles = histogram_to_quantiles(samples.flatten(0, 1), normalized, num_quantiles=n).reshape(5, 8, n)
            decoded = decode_quantiles(quantiles.flatten(0, 1), normalized, decoder="linear_cdf").reshape(5, 8, n)
            metadata = decoder_comparison_metadata(quantiles, "linear_cdf")
            artifact.update(samples=decoded, state_size=n, quantiles=quantiles,
                            decoder_comparison=metadata, quantile_sha256=metadata["quantile_sha256"])
            artifact["sampling_config"]["histogram_decoder"] = metadata["decoder_signature"]
            provenance.update(saved_quantiles=True, decoder_signature=metadata["decoder_signature"],
                              quantile_sha256=metadata["quantile_sha256"])
        _save_cache(legacy.prediction_path(experiment, "g5_k8", variant, dimensions, 256), artifact)
    return truth, truth_hash


class ReconstructionVisualizationTests(unittest.TestCase):
    def test_each_model_can_render_without_any_other_model_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in plots.MODEL_NAMES:
                with self.subTest(model=model):
                    experiment = root / model / "m64n128"
                    output = root / model / "figures"
                    truth, truth_hash = saved_fixture(experiment, models=(model,))
                    before = {path: file_sha256(path) for path in experiment.rglob("*") if path.is_file()}
                    # Render all five pages for one model; remaining model paths are absent.
                    with patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("No network")):
                        record = plots.main(["--experiment_dir", str(experiment), "--output_dir", str(output),
                                             "--models", model, "--num_conditions", "5", "--num_cpu_threads", "1"])
                    self.assertEqual(record["models"], [model])
                    self.assertEqual(set(record["sources"]), {model})
                    self.assertEqual(len(list(output.glob("*.png"))), 5)
                    self.assertEqual(len(list(output.glob("*.pdf"))), 5)
                    self.assertEqual(record["flow_nfe"], 256 if model in ("HFM", "QWFM") else None)
                    self.assertEqual(record, json.loads((output / "qualitative_provenance.json").read_text()))
                    self.assertEqual(before, {path: file_sha256(path) for path in before})
                    with np.load(output / "selected_display_arrays.npz") as arrays:
                        self.assertEqual(arrays[f"{model}_histogram"].shape, (5, 1 if model == "Regression" else 8, 128))
                        np.testing.assert_array_equal(arrays[f"{model}_shared_noise_signal"],
                            arrays[f"{model}_forward_signal"] + arrays["original_noise"][:, None, :])

    def test_comparison_works_for_other_settings_and_infers_qwfm_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            for dimensions in ((32, 128), (128, 256)):
                with self.subTest(dimensions=dimensions):
                    m, n = dimensions
                    experiment = Path(directory) / f"m{m}n{n}"
                    output = experiment / "visualizations"
                    saved_fixture(experiment, dimensions)
                    record = plots.main(["--experiment_dir", str(experiment), "--output_dir", str(output),
                                         "--models", *plots.MODEL_NAMES, "--num_conditions", "5", "--num_cpu_threads", "1"])
                    self.assertEqual(record["dimensions"], {"M": m, "N": n})
                    self.assertEqual(record["num_quantiles"], n)
                    self.assertEqual(set(record["sources"]), set(plots.MODEL_NAMES))
                    self.assertTrue(record["sources"]["QWFM"]["displayed_histograms_match_saved_quantiles"])
                    self.assertTrue(all(row["saved_draw_counts"] == {"HFM": 8, "QWFM": 8, "Regression": 1, "CVAE": 8}
                                        for row in record["figures"]))
                    with (output / "selected_cases.csv").open() as stream:
                        self.assertEqual([int(row["condition_id"]) for row in csv.DictReader(stream)], list(range(5)))

    def test_missing_selected_model_fails_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "m64n128"
            output = Path(directory) / "figures"
            saved_fixture(experiment, models=("HFM",))
            with self.assertRaises(FileNotFoundError):
                plots.main(["--experiment_dir", str(experiment), "--output_dir", str(output),
                            "--models", "HFM", "QWFM", "--num_conditions", "5"])
            self.assertFalse(output.exists())

    def test_single_qwfm_rejects_wrong_decoder_even_with_valid_file_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "m128n256"
            truth, digest = saved_fixture(experiment, (128, 256), models=("QWFM",))
            path = legacy.prediction_path(experiment, "g5_k8", "quantile_wasserstein_l256", (128, 256), 256)
            artifact = torch.load(path, weights_only=False)
            artifact["samples"] = artifact["samples"].roll(1, dims=-1)
            _save_cache(path, artifact)
            with self.assertRaisesRegex(ValueError, "do not match"):
                legacy.collect_display_arrays(truth, digest, legacy.select_cases(truth), experiment,
                                              256, 8, 256, models=("QWFM",))

    def test_duplicate_models_and_output_in_saved_inputs_are_rejected(self):
        with self.assertRaises(SystemExit):
            plots.parse_args(["--experiment_dir", "/tmp/m64n128", "--output_dir", "/tmp/figures",
                              "--models", "HFM", "HFM"])
        with self.assertRaises(SystemExit):
            plots.parse_args(["--experiment_dir", "/tmp/m64n128", "--output_dir",
                              "/tmp/m64n128/decoder_comparison/linear_cdf/proper_scores/g16384_k8/figures"])

    def test_direct_entrypoint_help_without_project_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", str(Path(plots.__file__).resolve()), "--help"],
                                    cwd=directory, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--models", result.stdout)


if __name__ == "__main__":
    unittest.main()
