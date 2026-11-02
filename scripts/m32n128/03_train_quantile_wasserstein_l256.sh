#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0,1,2,3"
NUM_PROCESSES="4"
NUM_MEASUREMENTS="32"
NUM_SPECTRUM_BINS="128"
L="256"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "${NUM_PROCESSES}" = "1" ]; then
  LAUNCHER=(python codes/train/train_quantile_wasserstein.py)
else
  LAUNCHER=(
    accelerate launch
    --multi_gpu
    --gpu_ids "${GPU_IDS}"
    --num_processes "${NUM_PROCESSES}"
    --main_process_port 0
    codes/train/train_quantile_wasserstein.py
  )
fi

OUTPUT_DIR="outputs/m32n128/quantile_wasserstein_m32_n128_l256_unet64x4_t1024"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${LAUNCHER[@]}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_quantiles="${L}" \
  --train_dataset_seed=20260810 \
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
  --train_batch_size=128 \
  --num_train_iterations=200000 \
  --log_every_iterations=10 \
  --dataloader_num_workers=0 \
  --eval_every_iterations=1000 \
  --num_fixed_eval_samples=128 \
  --fixed_eval_dataset_seed=20260812 \
  --eval_batch_size=32 \
  --eval_dataloader_num_workers=0 \
  --eval_samples_per_condition=2 \
  --eval_ode_steps=4 \
  --eval_num_visual_samples=2 \
  --eval_source_seed=161803 \
  --eval_metric_num_quantiles=128 \
  --eval_peak_relative_threshold=0.05 \
  --source_seed=271828 \
  --model_time_scale=1024 \
  --gradient_accumulation_batches=4 \
  --mixed_precision=fp16 \
  --learning_rate=1.0e-4 \
  --lr_scheduler=constant \
  --lr_warmup_iterations=0 \
  --lr_num_cycles=1 \
  --lr_power=1.0 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=1.0e-6 \
  --adam_epsilon=1.0e-8 \
  --max_grad_norm=1.0 \
  --use_ema \
  --ema_decay=0.9999 \
  --ema_min_decay=0.0 \
  --ema_update_after_iterations=0 \
  --ema_use_warmup \
  --ema_inv_gamma=1.0 \
  --ema_power=0.6666666666666666 \
  --checkpoint_every_iterations=1600 \
  --seed=42 \
  --resume_from_checkpoint=latest
