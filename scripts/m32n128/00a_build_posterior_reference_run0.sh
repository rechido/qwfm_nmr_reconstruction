#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0"
REFERENCE_RUN_ID="0"
NUM_MEASUREMENTS="32"
NUM_SPECTRUM_BINS="128"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_DIR="outputs/m32n128/reference_smc4096_v2"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" python codes/evaluate/build_posterior_reference.py \
  --output_dir="${OUTPUT_DIR}" \
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
  --num_test_conditions=32 \
  --test_dataset_seed=2026090301 \
  --reference_master_seed=2026090302 \
  --run_ids "${REFERENCE_RUN_ID}" \
  --device=cuda \
  --num_cpu_threads=2 \
  --num_particles=4096 \
  --output_samples=4096 \
  --target_conditional_ess_fraction=0.95 \
  --resample_ess_fraction=0.5 \
  --rejuvenation_sweeps=25 \
  --tempering_adaptation_fraction=0.5 \
  --final_rejuvenation_sweeps=10 \
  --maximum_final_rejuvenation_sweeps=128 \
  --final_diagnostic_interval=5 \
  --final_acceptance_window=5 \
  --final_minimum_mean_acceptance=0.10 \
  --final_minimum_unique_fraction=0.999 \
  --maximum_temperatures=512 \
  --beta_bisection_iterations=48 \
  --center_logit_step=0.15 \
  --log_cv_step_fraction=0.05 \
  --weight_logit_step=0.25 \
  --adaptation_target_acceptance=0.30 \
  --adaptation_gain=0.50 \
  --minimum_step_scale=1.0e-6 \
  --maximum_step_scale=2.0
