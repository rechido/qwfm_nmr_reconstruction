#!/usr/bin/env bash
set -euo pipefail

# Run this script without arguments. All experiment settings are defined here.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Level 3: compare all settings after their model pipelines are complete.
EXPERIMENTS_ROOT="outputs"
OUTPUT_DIR="outputs/visualizations/all_settings"
SETTINGS=("m32n128" "m64n128" "m128n256")
MODELS=("HFM" "QWFM" "Regression" "CVAE")
NFES=(1 4 16 64 256)

# Read completed reports only; save metric curves and CSV tables.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python codes/evaluate/visualize_results.py \
  --experiments_root="${EXPERIMENTS_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --settings "${SETTINGS[@]}" \
  --models "${MODELS[@]}" \
  --nfes "${NFES[@]}"
