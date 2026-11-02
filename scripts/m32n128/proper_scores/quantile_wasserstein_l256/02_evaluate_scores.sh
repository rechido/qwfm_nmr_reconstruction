#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="outputs/m32n128"
VARIANT="quantile_wasserstein_l256"
NUM_CONDITIONS="16384"
NUM_POSTERIOR_SAMPLES="8"
RUN_DIR="${EXPERIMENT_DIR}/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"
OUTPUT_DIR="${EXPERIMENT_DIR}/metric_comparisons/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# CPU ES/KS only; no network inference, SMC, PRDC, or sweep resampling.
# Merge compatible previously evaluated models into comparison tables and figures.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python codes/evaluate/evaluate_proper_scores.py \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --variant="${VARIANT}" \
  --run_dir="${RUN_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_conditions="${NUM_CONDITIONS}" \
  --sample_count="${NUM_POSTERIOR_SAMPLES}" \
  --num_cpu_threads=2
