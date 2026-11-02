#!/usr/bin/env bash
set -euo pipefail

# Run this script without arguments. All experiment settings are defined here.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

GPU_IDS="0"
NUM_PROCESSES="1"
NUM_MEASUREMENTS="128"
NUM_SPECTRUM_BINS="256"
LATENT_DIM="16"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ "${NUM_PROCESSES}" = "1" ]; then
  LAUNCHER=(python codes/evaluate/evaluate_conditional_vae.py)
else
  LAUNCHER=(
    accelerate launch
    --multi_gpu
    --gpu_ids "${GPU_IDS}"
    --num_processes "${NUM_PROCESSES}"
    --main_process_port 0
    codes/evaluate/evaluate_conditional_vae.py
  )
fi

OUTPUT_DIR="outputs/m128n256/conditional_vae_m128_n256_unet64x4_z16"
PRIMARY_REFERENCE="outputs/m128n256/reference_smc4096_v2/run_00/reference_run_00.pt"
ORACLE_REFERENCE="outputs/m128n256/reference_smc4096_v2/run_01/reference_run_01.pt"
python codes/evaluate/check_paper_checkpoint.py \
  --training_dir="${OUTPUT_DIR}" \
  --checkpoint_path=latest \
  --required_iteration=200000 \
  --reference_audit="outputs/m128n256/reference_smc4096_v2/audit/reference_audit_summary.json"

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
  --reuse_existing=true \
  --peak_relative_threshold=0.05 \
  --latent_dim="${LATENT_DIM}" \
  --encoder_hidden_dim=256 \
  --encoder_num_layers=4 \
  --max_decode_batch_size=512 \
  --sample_seed=161803 \
  --mixed_precision=fp16 \
  --seed=42
