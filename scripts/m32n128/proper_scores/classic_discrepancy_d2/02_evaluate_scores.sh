#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

EXPERIMENT_DIR="outputs/m32n128"
VARIANT="classic_discrepancy_d2"
NUM_CONDITIONS="16384"
NUM_POSTERIOR_SAMPLES="8"
RUN_DIR="${EXPERIMENT_DIR}/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"
OUTPUT_DIR="${EXPERIMENT_DIR}/metric_comparisons/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python codes/evaluate/evaluate_proper_scores.py \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --run_dir="${RUN_DIR}" \
  --variant="${VARIANT}" \
  --num_conditions="${NUM_CONDITIONS}" \
  --sample_count="${NUM_POSTERIOR_SAMPLES}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_cpu_threads=2
