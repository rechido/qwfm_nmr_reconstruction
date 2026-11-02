#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0"
NUM_PROCESSES="1"
NUM_MEASUREMENTS="64"
NUM_SPECTRUM_BINS="128"
L="128"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "${NUM_PROCESSES}" = "1" ]; then
  LAUNCHER=(python codes/evaluate/evaluate_quantile_wasserstein.py)
else
  LAUNCHER=(
    accelerate launch
    --multi_gpu
    --gpu_ids "${GPU_IDS}"
    --num_processes "${NUM_PROCESSES}"
    --main_process_port 0
    codes/evaluate/evaluate_quantile_wasserstein.py
  )
fi

OUTPUT_DIR="outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024"
PRIMARY_REFERENCE="outputs/m64n128/reference_smc4096_v2/run_00/reference_run_00.pt"
ORACLE_REFERENCE="outputs/m64n128/reference_smc4096_v2/run_01/reference_run_01.pt"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${LAUNCHER[@]}" \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoint_path=latest \
  --model_subfolder=unet_ema \
  --eval_output_subdir=evaluation \
  --primary_reference="${PRIMARY_REFERENCE}" \
  --oracle_reference="${ORACLE_REFERENCE}" \
  --num_measurements="${NUM_MEASUREMENTS}" \
  --num_spectrum_bins="${NUM_SPECTRUM_BINS}" \
  --rate_minimum=2.0 \
  --rate_maximum=250.0 \
  --echo_time_minimum=0.008 \
  --echo_time_maximum=0.512 \
  --noise_standard_deviation=0.005 \
  --component_count_minimum=1 \
  --component_count_maximum=5 \
  --center_minimum=4.0 \
  --center_maximum=160.0 \
  --coefficient_of_variation_minimum=0.04 \
  --coefficient_of_variation_maximum=0.25 \
  --mixture_weight_concentration=2.0 \
  --minimum_component_retained_mass=0.995 \
  --maximum_prior_attempts=100000 \
  --eval_batch_size=4 \
  --dataloader_num_workers=0 \
  --num_posterior_samples=256 \
  --num_visual_samples=16 \
  --reference_repeats=20 \
  --nearest_k=3 \
  --w2_num_quantiles=2048 \
  --distance_chunk_size=64 \
  --metric_seed=2026090304 \
  --credible_interval_level=0.90 \
  --reuse_existing=false \
  --peak_relative_threshold=0.05 \
  --num_quantiles="${L}" \
  --eval_ode_steps=1,4,16,64,256 \
  --source_seed=161803 \
  --model_time_scale=1024 \
  --mixed_precision=fp16 \
  --seed=42
