"""Check the final EMA checkpoint before a paper sampling job starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "codes"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from utils.predictive_model_utils import inspect_training_model


def validate_reference_audit(path: Path, specification_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SMC audit: {path}. Run 13_audit_references.sh before PR evaluation.")
    audit = json.loads(path.read_text())
    if (audit.get("audit_protocol_version") != "independent_smc_reference_audit_v3"
            or audit.get("overall_status") not in ("pass", "warn")
            or audit.get("specification_hash") != specification_hash
            or audit.get("reference_run_ids") != [0, 1, 2]
            or audit.get("condition_count") != 32):
        raise ValueError(f"{path}: missing, rejected, or incompatible paper SMC audit. Check the reference-build and audit stages.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_path", default="latest")
    parser.add_argument("--required_iteration", type=int, default=200000)
    parser.add_argument("--reference_audit", type=Path, help="Required completed SMC audit for a PR job.")
    args = parser.parse_args(argv)
    if args.required_iteration <= 0:
        parser.error("required_iteration must be positive.")
    info = inspect_training_model(args.training_dir, args.checkpoint_path, "unet_ema")
    iteration = info["checkpoint_state"].get("global_iteration")
    if iteration != args.required_iteration:
        raise ValueError(
            f"Paper sampling requires the final EMA checkpoint at iteration "
            f"{args.required_iteration}, but {args.training_dir} contains iteration "
            f"{iteration}. Finish the training script before sampling."
        )
    if args.reference_audit is not None:
        validate_reference_audit(args.reference_audit, info["spec"].specification_hash)
    print(f"Final EMA checkpoint verified: {info['checkpoint']} (iteration {iteration})")


if __name__ == "__main__":
    main()
