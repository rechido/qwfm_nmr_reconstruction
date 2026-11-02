#!/usr/bin/env bash
set -euo pipefail

# Run this script without arguments. All experiment settings are defined here.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

EXPERIMENT_DIR="outputs/m32n128"
VARIANT="histogram_linear"
NUM_CONDITIONS="16384"
NUM_POSTERIOR_SAMPLES="8"
RUN_DIR="${EXPERIMENT_DIR}/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"
OUTPUT_DIR="${EXPERIMENT_DIR}/metric_comparisons/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Read saved samples on CPU. Export the final paper figures in the last stage.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python codes/evaluate/evaluate_proper_scores.py \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --variant="${VARIANT}" \
  --run_dir="${RUN_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_conditions="${NUM_CONDITIONS}" \
  --sample_count="${NUM_POSTERIOR_SAMPLES}" \
  --ground_metrics l2 w2 \
  --num_cpu_threads=2 \
  --skip_plots
