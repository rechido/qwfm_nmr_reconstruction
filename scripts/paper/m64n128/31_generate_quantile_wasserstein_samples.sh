#!/usr/bin/env bash
set -euo pipefail

# Run this script without arguments. All experiment settings are defined here.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

GPU_IDS="0,1,2,3"
NUM_PROCESSES="4"
EXPERIMENT_DIR="outputs/m64n128"
VARIANT="quantile_wasserstein_l128"
NUM_CONDITIONS="16384"
NUM_POSTERIOR_SAMPLES="8"
EVAL_ODE_STEPS="1,4,16,64,256"
RUN_DIR="${EXPERIMENT_DIR}/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"
COMPARISON_ROOT="${EXPERIMENT_DIR}/decoder_comparison"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Read the existing fixed test split; do not prepare a new test set.
# Generate terminal quantiles once and save paired hard-bin / linear-CDF histograms.
# Original production predictions are untouched. Four GPUs share the conditions.
python codes/evaluate/check_paper_checkpoint.py \
  --training_dir="outputs/m64n128/quantile_wasserstein_m64_n128_l128_unet64x4_t1024" \
  --checkpoint_path=latest \
  --required_iteration=200000

if [ "${NUM_PROCESSES}" = "1" ]; then
  LAUNCHER=(python codes/evaluate/generate_qwfm_decoder_comparison.py)
else
  LAUNCHER=(accelerate launch --multi_gpu --gpu_ids "${GPU_IDS}"
    --num_processes "${NUM_PROCESSES}" --main_process_port 0
    codes/evaluate/generate_qwfm_decoder_comparison.py)
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${LAUNCHER[@]}" \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --variant="${VARIANT}" \
  --run_dir="${RUN_DIR}" \
  --comparison_root="${COMPARISON_ROOT}" \
  --num_conditions="${NUM_CONDITIONS}" \
  --num_posterior_samples="${NUM_POSTERIOR_SAMPLES}" \
  --eval_ode_steps="${EVAL_ODE_STEPS}" \
  --eval_batch_size=64 \
  --max_generated_batch_size=512 \
  --max_decode_batch_size=1024 \
  --mixed_precision=fp16 \
  --test_dataset_seed=2026092901 \
  --source_seed=2026092902 \
  --reuse_existing=true \
  --dataloader_num_workers=0 \
  --num_cpu_threads=2 \
  --checkpoint_path=latest \
  --model_subfolder=unet_ema \
  --progress_every=20
