#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0"
NUM_PROCESSES="1"
NUM_MEASUREMENTS="32"
NUM_SPECTRUM_BINS="128"
NUM_SOLVER_WORKERS="8"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "${NUM_PROCESSES}" = "1" ]; then
  LAUNCHER=(python codes/evaluate/evaluate_classical_baseline.py)
else
  LAUNCHER=(
    accelerate launch
    --multi_gpu
    --gpu_ids "${GPU_IDS}"
    --num_processes "${NUM_PROCESSES}"
    --main_process_port 0
    codes/evaluate/evaluate_classical_baseline.py
  )
fi


OUTPUT_DIR="outputs/m32n128/classical_nnls_m32_n128"
PRIMARY_REFERENCE="outputs/m32n128/reference_smc4096_v2/run_00/reference_run_00.pt"
ORACLE_REFERENCE="outputs/m32n128/reference_smc4096_v2/run_01/reference_run_01.pt"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${LAUNCHER[@]}" \
  --reference_path="${PRIMARY_REFERENCE}" \
  --output_dir="${OUTPUT_DIR}/evaluation" \
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
  --method=nnls \
  --bayes_beta_source=official_nnls_residual \
  --num_samples_per_condition=256 \
  --num_workers="${NUM_SOLVER_WORKERS}" \
  --x2_factor=1.02 \
  --regularization_minimum=1.0e-8 \
  --optimizer_tolerance=1.0e-5 \
  --optimizer_max_iterations=300 \
  --nnls_max_iterations=0 \
  --reuse_existing

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python codes/evaluate/evaluate_saved_samples.py \
  --samples="${OUTPUT_DIR}/evaluation/samples.pt" \
  --primary_reference="${PRIMARY_REFERENCE}" \
  --oracle_reference="${ORACLE_REFERENCE}" \
  --output_dir="${OUTPUT_DIR}/evaluation/metrics" \
  --sample_count=256 \
  --reference_repeats=20 \
  --nearest_k=3 \
  --w2_num_quantiles=2048 \
  --distance_chunk_size=64 \
  --seed=2026090304 \
  --credible_interval_level=0.90 \
  --device=cuda \
  --overwrite
