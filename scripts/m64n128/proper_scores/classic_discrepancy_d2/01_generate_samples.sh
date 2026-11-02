#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

EXPERIMENT_DIR="outputs/m64n128"
VARIANT="classic_discrepancy_d2"
NUM_CONDITIONS="16384"
NUM_POSTERIOR_SAMPLES="8"
RUN_DIR="${EXPERIMENT_DIR}/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"
NUM_SOLVER_WORKERS="8"

# First run proper_scores/00_prepare_test_conditions.sh for this experiment.
# This CPU solver only reads its shared test pairs; no learned model is required.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python codes/evaluate/generate_proper_score_classical.py \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --variant="${VARIANT}" \
  --run_dir="${RUN_DIR}" \
  --num_conditions="${NUM_CONDITIONS}" \
  --num_posterior_samples="${NUM_POSTERIOR_SAMPLES}" \
  --test_dataset_seed=2026092901 \
  --num_workers="${NUM_SOLVER_WORKERS}" \
  --condition_batch_size=16 \
  --progress_every=20 \
  --progress_interval_seconds=30 \
  --reuse_existing
