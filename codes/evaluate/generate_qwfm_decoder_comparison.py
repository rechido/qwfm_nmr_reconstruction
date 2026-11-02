"""Generate QWFM quantiles once, preserving naive and linear-CDF decodings.

All outputs are isolated from the existing production samples. The original
fixed test split is linked read-only and is never regenerated.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "codes"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate import generate_proper_score_samples as generation
from evaluate.decode_qwfm_samples import main as decode_saved
from utils.quantile_decoder_artifacts import reuse_test_conditions


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--comparison_root", type=Path)
    extra.add_argument("--source_run_dir", type=Path)
    extra.add_argument("--decode_only", action="store_true",
                       help="Re-decode saved final quantiles on CPU; no checkpoint loading or inference.")
    options, remaining = extra.parse_known_args(argv)
    if "--help" in argv or "-h" in argv:
        extra.print_help()
    args = generation.parse_args(remaining)
    if not args.variant.startswith("quantile_wasserstein_l"):
        raise ValueError("Decoder comparison accepts QWFM variants only.")
    root = (options.comparison_root or args.experiment_dir / "decoder_comparison").resolve()
    source = (options.source_run_dir or args.run_dir).resolve()
    name = f"g{args.num_conditions}_k{args.num_posterior_samples}"
    hard_run = root / "hard_bin" / "proper_scores" / name
    linear_run = root / "linear_cdf" / "proper_scores" / name
    if any(source == target or source in target.parents or target in source.parents
           for target in (hard_run, linear_run)):
        raise ValueError("Original test run and comparison runs must occupy separate directory trees.")
    # Distributed ranks serialize this tiny initialization only. Subsequent
    # generation retains the existing four-GPU partition and seed contracts.
    reuse_test_conditions(source, hard_run)
    reuse_test_conditions(source, linear_run)
    if not options.decode_only:
        generation.main([*remaining, "--run_dir", str(hard_run), "--save_quantiles", "true"])
    if int(os.environ.get("RANK", "0")) == 0:
        decode_saved([
            "--source_run_dir", str(hard_run), "--output_run_dir", str(linear_run),
            "--variant", args.variant, "--num_conditions", str(args.num_conditions),
            "--test_dataset_seed", str(args.test_dataset_seed),
            "--num_cpu_threads", str(args.num_cpu_threads),
        ])


if __name__ == "__main__":
    main()
