#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python codes/evaluate/plot_metric_comparison.py \
  --evaluations \
    "HFM-NFE1=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0001/metrics/evaluation_summary.json" \
    "QWFM-L128-NFE1=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0001/metrics/evaluation_summary.json" \
    "QWFM-L256-NFE1=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0001/metrics/evaluation_summary.json" \
    "QWFM-L512-NFE1=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0001/metrics/evaluation_summary.json" \
    "HFM-NFE4=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0004/metrics/evaluation_summary.json" \
    "QWFM-L128-NFE4=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0004/metrics/evaluation_summary.json" \
    "QWFM-L256-NFE4=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0004/metrics/evaluation_summary.json" \
    "QWFM-L512-NFE4=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0004/metrics/evaluation_summary.json" \
    "HFM-NFE16=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0016/metrics/evaluation_summary.json" \
    "QWFM-L128-NFE16=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0016/metrics/evaluation_summary.json" \
    "QWFM-L256-NFE16=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0016/metrics/evaluation_summary.json" \
    "QWFM-L512-NFE16=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0016/metrics/evaluation_summary.json" \
    "HFM-NFE64=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0064/metrics/evaluation_summary.json" \
    "QWFM-L128-NFE64=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0064/metrics/evaluation_summary.json" \
    "QWFM-L256-NFE64=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0064/metrics/evaluation_summary.json" \
    "QWFM-L512-NFE64=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0064/metrics/evaluation_summary.json" \
    "HFM-NFE256=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L128-NFE256=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L256-NFE256=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L512-NFE256=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "Regression-MSE=outputs/m64n128/deterministic_regression_m64_n128_unet64x4/evaluation/metrics/evaluation_summary.json" \
    "Regression-MSE-W1=outputs/m64n128/deterministic_regression_mse_w1_m64_n128_unet64x4/evaluation/metrics/evaluation_summary.json" \
    "CVAE-z16=outputs/m64n128/conditional_vae_m64_n128_unet64x4_z16/evaluation/metrics/evaluation_summary.json" \
  --plot_kind=nfe_sweep \
  --output_dir="outputs/m64n128/metric_comparisons/fm_nfe_sweep" \
  --title="FM NFE sweep" \
  --filename_prefix=fm_nfe_sweep \
  --dpi=180 \
  --skip_missing \
  --overwrite
