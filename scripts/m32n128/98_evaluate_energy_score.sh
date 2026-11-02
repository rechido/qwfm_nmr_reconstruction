#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Energy + Gaussian kernel scores from cached samples: no checkpoint loading, network inference, or SMC sampling.
# Missing test samples / measured truth are reported, never synthesized.
# Kernel bandwidth uses a separate fixed simulator calibration, shared by all models.
python codes/evaluate/evaluate_energy_score.py \
  --experiment_dir="outputs/m32n128" \
  --num_cpu_threads=2 \
  --distance_chunk_size=64 \
  --update_existing_metrics

python codes/evaluate/plot_energy_score.py \
  --summary="outputs/m32n128/metric_comparisons/energy_score/energy_score_summary.json"

python codes/evaluate/plot_energy_score.py \
  --score=kernel_score \
  --summary="outputs/m32n128/metric_comparisons/kernel_score/kernel_score_summary.json"
