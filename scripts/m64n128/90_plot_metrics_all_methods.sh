#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python codes/evaluate/plot_metric_comparison.py \
  --evaluations \
    "HFM=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L128=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L256=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "QWFM-L512=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0256/metrics/evaluation_summary.json" \
    "Regression-MSE=outputs/m64n128/deterministic_regression_m64_n128_unet64x4/evaluation/metrics/evaluation_summary.json" \
    "Regression-MSE-W1=outputs/m64n128/deterministic_regression_mse_w1_m64_n128_unet64x4/evaluation/metrics/evaluation_summary.json" \
    "CNF=outputs/m64n128/conditional_normalizing_flow_m64_n128_realnvp8x256/evaluation/metrics/evaluation_summary.json" \
    "CVAE-z16=outputs/m64n128/conditional_vae_m64_n128_unet64x4_z16/evaluation/metrics/evaluation_summary.json" \
    "CVAE-z32=outputs/m64n128/conditional_vae_m64_n128_unet64x4_z32/evaluation/metrics/evaluation_summary.json" \
    "NNLS=outputs/m64n128/classical_nnls_m64_n128/evaluation/metrics/evaluation_summary.json" \
    "X2-D2=outputs/m64n128/classical_x2_d2_m64_n128/evaluation/metrics/evaluation_summary.json" \
    "Discrepancy-D2=outputs/m64n128/classical_discrepancy_d2_m64_n128/evaluation/metrics/evaluation_summary.json" \
    "BayesReg-I-official=outputs/m64n128/classical_bayesreg_i_official_nnls_residual_m64_n128/evaluation/metrics/evaluation_summary.json" \
    "BayesReg-I-known-noise=outputs/m64n128/classical_bayesreg_i_known_noise_m64_n128/evaluation/metrics/evaluation_summary.json" \
  --output_dir="outputs/m64n128/metric_comparisons/all_methods" \
  --title="All methods (FM: NFE=256)" \
  --filename_prefix=all_methods \
  --dpi=180 \
  --skip_missing \
  --overwrite
