#!/usr/bin/env bash
set -euo pipefail

# Run this script without arguments. All experiment settings are defined here.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

REFERENCE_DIR="outputs/m128n256/reference_smc4096_v2"
python codes/evaluate/audit_reference.py \
  --references \
    "${REFERENCE_DIR}/run_00/reference_run_00.pt" \
    "${REFERENCE_DIR}/run_01/reference_run_01.pt" \
    "${REFERENCE_DIR}/run_02/reference_run_02.pt" \
  --output_dir="${REFERENCE_DIR}/audit" \
  --sample_count=256 \
  --num_cpu_threads=2 \
  --peak_count_tv_pass=0.05 \
  --peak_count_tv_warn=0.10 \
  --w2_num_quantiles=2048 \
  --distance_chunk_size=64 \
  --seed=2026090305 \
  --representation_quantiles=128 \
  --suitability_lower_quantile=0.25 \
  --minimum_w2_spread_pass=0.05 \
  --minimum_w2_spread_warn=0.02 \
  --minimum_w2_floor_ratio_pass=5.0 \
  --minimum_w2_floor_ratio_warn=2.0 \
  --overwrite \
  --fail_on_rejection
