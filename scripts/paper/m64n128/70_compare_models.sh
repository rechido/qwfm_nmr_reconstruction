#!/usr/bin/env bash
set -euo pipefail

# Run without arguments. Edit the settings below to customize this visualization.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SETTING="m64n128"
EXPERIMENT_DIR="outputs/${SETTING}"
OUTPUT_DIR="${EXPERIMENT_DIR}/visualizations/comparison"
MODELS=("HFM" "QWFM" "Regression" "CVAE")
NFES=(1 4 16 64 256)
RECONSTRUCTION_NFE=256
NUM_QUANTILES=128
NUM_CONDITIONS=16384
SAMPLE_COUNT=8
NUM_CPU_THREADS=2

# Compare the four completed models in this setting; no other setting is required.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python codes/evaluate/visualize_results.py \
  --experiments_root=outputs \
  --output_dir="${OUTPUT_DIR}/metrics" \
  --settings "${SETTING}" \
  --models "${MODELS[@]}" \
  --nfes "${NFES[@]}"

# Reuse saved spectra for decay/spectrum previews, without neural inference.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python codes/evaluate/visualize_reconstructions.py \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --output_dir="${OUTPUT_DIR}/reconstructions" \
  --models "${MODELS[@]}" \
  --nfe="${RECONSTRUCTION_NFE}" \
  --num_conditions="${NUM_CONDITIONS}" \
  --sample_count="${SAMPLE_COUNT}" \
  --num_quantiles="${NUM_QUANTILES}" \
  --num_cpu_threads="${NUM_CPU_THREADS}"
