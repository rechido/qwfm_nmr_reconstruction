from __future__ import annotations

import contextlib
import fcntl
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from codes.data import ContinuousGammaMixtureDataset, ContinuousNMRSpec  # noqa: E402
from codes.evaluate.generate_proper_score_classical import CLASSICAL_VARIANTS, main  # noqa: E402
from codes.evaluate.generate_proper_score_samples import PROTOCOL, TEST_SPLIT  # noqa: E402
from codes.utils.io_utils import load_torch_artifact  # noqa: E402
from codes.utils.predictive_generation_utils import _save_cache  # noqa: E402


class ClassicalProperScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        spec = ContinuousNMRSpec()
        dataset = ContinuousGammaMixtureDataset(
            num_samples=2, spec=spec, master_seed=2026092901,
            split=TEST_SPLIT, refresh_each_epoch=False,
        )
        cls.truth = default_collate([dataset[index] for index in range(2)])
        cls.truth.update(
            protocol_id=spec.protocol_id, specification=spec.to_dict(),
            specification_hash=spec.specification_hash, split=TEST_SPLIT,
            master_seed=2026092901, grid=dataset.grid.as_artifact(),
            proper_score_provenance={
                "protocol_version": PROTOCOL,
                "num_conditions": 2,
                "specification_hash": spec.specification_hash,
                "master_seed": 2026092901,
                "split": TEST_SPLIT,
            },
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.experiment = Path(self.temporary.name) / "m32n128"
        self.run = self.experiment / "proper_scores" / "g2_k8"
        _save_cache(self.run / "test_conditions.pt", self.truth)
        self.args = [
            "--experiment_dir", str(self.experiment),
            "--num_conditions", "2", "--num_posterior_samples", "8",
            "--num_workers", "1", "--condition_batch_size", "1",
        ]

    def run_generator(self, variant: str, *extra: str) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main([*self.args, "--variant", variant, *extra])
        return output.getvalue()

    def test_all_solvers_store_one_point_with_shared_conditions_and_raw_amplitudes(self) -> None:
        for variant, (method, beta_source) in CLASSICAL_VARIANTS.items():
            with self.subTest(variant=variant):
                log = self.run_generator(variant)
                self.assertIn("eta=", log)
                artifact = load_torch_artifact(self.run / variant / "samples.pt")
                self.assertEqual(tuple(artifact["samples"].shape), (2, 1, 128))
                self.assertTrue(artifact["deterministic"])
                self.assertEqual(artifact["num_samples_per_condition"], 1)
                torch.testing.assert_close(artifact["condition_ids"], self.truth["sample_id"])
                torch.testing.assert_close(artifact["measurements"], self.truth["measurement"])
                torch.testing.assert_close(artifact["samples"].sum(-1), torch.ones(2, 1))
                self.assertEqual(artifact["proper_score_provenance"]["requested_sample_count"], 8)
                self.assertEqual(artifact["proper_score_provenance"]["stored_sample_count"], 1)
                self.assertEqual(artifact["classical_results"]["method"], method)
                self.assertEqual(artifact["classical_results"]["bayes_beta_source"],
                                 beta_source if method == "bayesreg_i" else None)
                torch.testing.assert_close(
                    artifact["forward_spectra"][:, 0],
                    torch.from_numpy(artifact["classical_results"]["raw_spectra"]).float(),
                )

    def test_reuse_checks_identity_and_refuses_silent_overwrite(self) -> None:
        self.run_generator("classic_nnls")
        with self.assertRaises(FileExistsError):
            self.run_generator("classic_nnls")
        self.assertIn("Reused checked", self.run_generator("classic_nnls", "--reuse_existing"))
        with self.assertRaisesRegex(ValueError, "sampling settings|provenance|settings"):
            self.run_generator("classic_nnls", "--reuse_existing", "--optimizer_tolerance", "0.01")

    def test_missing_or_modified_shared_truth_is_not_silently_regenerated(self) -> None:
        path = self.run / "test_conditions.pt"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "00_prepare_test_conditions"):
            self.run_generator("classic_nnls")
        path.write_bytes(b"not the shared artifact")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_generator("classic_nnls")

    def test_wrong_test_seed_is_rejected_before_solving(self) -> None:
        with self.assertRaises(ValueError):
            self.run_generator("classic_nnls", "--test_dataset_seed", "42")
        self.assertFalse((self.run / "classic_nnls" / "samples.pt").exists())

    def test_old_exclusive_generator_is_rejected_before_writing(self) -> None:
        with (self.run / ".generation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "older generator"):
                self.run_generator("classic_nnls")
        self.assertFalse((self.run / "classic_nnls" / "samples.pt").exists())


if __name__ == "__main__":
    unittest.main()
