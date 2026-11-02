#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python codes/evaluate/visualize_posterior_samples.py \
  --reference="outputs/m64n128/reference_smc4096_v2/run_00/reference_run_00.pt" \
  --models \
    "HFM=outputs/m64n128/histogram_linear_m64_n128_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
    "QWFM-L128=outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
    "QWFM-L256=outputs/m64n128/quantile_wasserstein_m64_n128_l256_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
    "QWFM-L512=outputs/m64n128/quantile_wasserstein_m64_n128_l512_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
    "Regression-MSE=outputs/m64n128/deterministic_regression_m64_n128_unet64x4/evaluation/samples.pt" \
    "Regression-MSE-W1=outputs/m64n128/deterministic_regression_mse_w1_m64_n128_unet64x4/evaluation/samples.pt" \
    "CNF=outputs/m64n128/conditional_normalizing_flow_m64_n128_realnvp8x256/evaluation/samples.pt" \
    "CVAE-z16=outputs/m64n128/conditional_vae_m64_n128_unet64x4_z16/evaluation/samples.pt" \
    "CVAE-z32=outputs/m64n128/conditional_vae_m64_n128_unet64x4_z32/evaluation/samples.pt" \
    "NNLS=outputs/m64n128/classical_nnls_m64_n128/evaluation/samples.pt" \
    "X2-D2=outputs/m64n128/classical_x2_d2_m64_n128/evaluation/samples.pt" \
    "Discrepancy-D2=outputs/m64n128/classical_discrepancy_d2_m64_n128/evaluation/samples.pt" \
    "BayesReg-I-official=outputs/m64n128/classical_bayesreg_i_official_nnls_residual_m64_n128/evaluation/samples.pt" \
    "BayesReg-I-known-noise=outputs/m64n128/classical_bayesreg_i_known_noise_m64_n128/evaluation/samples.pt" \
  --output_dir="outputs/m64n128/qualitative/all_methods" \
  --dpi=180 \
  --nfe_values 1 4 16 64 256 \
  --skip_missing \
  --overwrite
