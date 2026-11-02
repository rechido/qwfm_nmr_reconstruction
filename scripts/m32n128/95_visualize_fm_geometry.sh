#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python codes/evaluate/visualize_posterior_samples.py \
  --reference="outputs/m32n128/reference_smc4096_v2/run_00/reference_run_00.pt" \
  --models \
    "HFM=outputs/m32n128/histogram_linear_m32_n128_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
    "QWFM-L128=outputs/m32n128/quantile_wasserstein_m32_n128_l128_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
    "QWFM-L256=outputs/m32n128/quantile_wasserstein_m32_n128_l256_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
    "QWFM-L512=outputs/m32n128/quantile_wasserstein_m32_n128_l512_unet64x4_t1024/evaluation/nfe_0256/samples.pt" \
  --output_dir="outputs/m32n128/qualitative/fm_geometry" \
  --dpi=180 \
  --nfe_values 1 4 16 64 256 \
  --skip_missing \
  --overwrite
