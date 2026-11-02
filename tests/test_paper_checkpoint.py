"""Reject incomplete training and rejected/mismatched reference banks early."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codes.evaluate import check_paper_checkpoint as check


class PaperCheckpointTests(unittest.TestCase):
    def test_incomplete_training_is_rejected_before_sampling(self):
        model = {"checkpoint_state": {"global_iteration": 199999}}
        with patch.object(check, "inspect_training_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "Finish the training"):
                check.main(["--training_dir", "/unused", "--required_iteration", "200000"])

    def test_final_checkpoint_and_nonrejected_matching_audit_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            model = {"checkpoint_state": {"global_iteration": 200000},
                     "spec": SimpleNamespace(specification_hash="test-spec"),
                     "checkpoint": Path(directory) / "checkpoint"}
            for status in ("pass", "warn"):
                audit.write_text(json.dumps({"audit_protocol_version": "independent_smc_reference_audit_v3",
                                             "overall_status": status, "specification_hash": "test-spec",
                                             "reference_run_ids": [0, 1, 2], "condition_count": 32}))
                with patch.object(check, "inspect_training_model", return_value=model) as inspect:
                    check.main(["--training_dir", directory, "--reference_audit", str(audit)])
                    inspect.assert_called_once_with(Path(directory), "latest", "unet_ema")

    def test_failed_or_different_reference_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            base = {"audit_protocol_version": "independent_smc_reference_audit_v3",
                    "overall_status": "pass", "specification_hash": "test-spec",
                    "reference_run_ids": [0, 1, 2], "condition_count": 32}
            for change in ({"overall_status": "fail"}, {"specification_hash": "different"},
                           {"reference_run_ids": [0, 1]}, {"condition_count": 31}):
                audit.write_text(json.dumps({**base, **change}))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    check.validate_reference_audit(audit, "test-spec")

    def test_missing_audit_explains_the_required_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "13_audit_references.sh"):
                check.validate_reference_audit(Path(directory) / "missing.json", "test-spec")


if __name__ == "__main__":
    unittest.main()
