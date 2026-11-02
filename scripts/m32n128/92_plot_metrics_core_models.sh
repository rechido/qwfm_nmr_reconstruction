#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python codes/evaluate/plot_metric_comparison.py \
  --evaluations \
    "Regression-MSE=outputs/m32n128/deterministic_regression_m32_n128_unet64x4/evaluation/metrics/evaluation_summary.json" \
    "Regression-MSE-W1=outputs/m32n128/deterministic_regression_mse_w1_m32_n128_unet64x4/evaluation/metrics/evaluation_summary.json" \
    "CNF=outputs/m32n128/conditional_normalizing_flow_m32_n128_realnvp8x256/evaluation/metrics/evaluation_summary.json" \
    "CVAE-z16=outputs/m32n128/conditional_vae_m32_n128_unet64x4_z16/evaluation/metrics/evaluation_summary.json" \
    "CVAE-z32=outputs/m32n128/conditional_vae_m32_n128_unet64x4_z32/evaluation/metrics/evaluation_summary.json" \
    "HFM=outputs/m32n128/histogram_linear_m32_n128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L128=outputs/m32n128/quantile_wasserstein_m32_n128_l128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
  --output_dir="outputs/m32n128/metric_comparisons/core_models" \
  --title="Core learned-model comparison (FM: NFE=256)" \
  --filename_prefix=core_models \
  --dpi=180 \
  --skip_missing \
  --overwrite
