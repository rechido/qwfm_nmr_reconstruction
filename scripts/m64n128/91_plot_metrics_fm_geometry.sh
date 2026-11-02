#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python codes/evaluate/plot_metric_comparison.py \
  --evaluations \
    "HFM=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L128=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L256=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L512=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
  --output_dir="outputs/m64n128/metric_comparisons/fm_geometry" \
  --title="FM geometry and representation (NFE=256)" \
  --filename_prefix=fm_geometry \
  --dpi=180 \
  --skip_missing \
  --overwrite
