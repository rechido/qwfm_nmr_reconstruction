#!/usr/bin/env bash
set -euo pipefail

# Run this script without arguments. All experiment settings are defined here.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

EXPERIMENT_DIR="outputs/m128n256"
NUM_CONDITIONS="16384"
NUM_POSTERIOR_SAMPLES="8"
NUM_MEASUREMENTS="128"
NUM_SPECTRUM_BINS="256"
RUN_DIR="${EXPERIMENT_DIR}/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Prepare common simulator (y, h) pairs once, without a model or checkpoint.
# A compatible existing test_conditions.pt is validated and reused unchanged.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python codes/evaluate/prepare_proper_score_conditions.py \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --run_dir="${RUN_DIR}" \
  --num_conditions="${NUM_CONDITIONS}" \
  --num_posterior_samples="${NUM_POSTERIOR_SAMPLES}" \
  --num_measurements="${NUM_MEASUREMENTS}" \
  --num_spectrum_bins="${NUM_SPECTRUM_BINS}" \
  --test_dataset_seed=2026092901 \
  --num_cpu_threads=2
