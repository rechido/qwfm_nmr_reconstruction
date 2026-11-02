#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0,1,2,3"
NUM_PROCESSES="4"
EXPERIMENT_DIR="outputs/m64n128"
VARIANT="histogram_linear"
NUM_CONDITIONS="16384"
NUM_POSTERIOR_SAMPLES="8"
EVAL_ODE_STEPS="1,4,16,64,256"
RUN_DIR="${EXPERIMENT_DIR}/proper_scores/g${NUM_CONDITIONS}_k${NUM_POSTERIOR_SAMPLES}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# First run proper_scores/00_prepare_test_conditions.sh for this experiment.
# Read its fixed (y, h) pairs; different models may generate concurrently.
# Four GPUs generate exactly one quarter of the conditions each. No PRDC or SMC.
accelerate launch \
  --multi_gpu \
  --gpu_ids "${GPU_IDS}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port 0 \
  codes/evaluate/generate_proper_score_samples.py \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --variant="${VARIANT}" \
  --run_dir="${RUN_DIR}" \
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
