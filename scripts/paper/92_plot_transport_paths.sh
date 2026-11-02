#!/usr/bin/env bash
set -euo pipefail

# Run this script without arguments. All experiment settings are defined here.
if (( $# != 0 )); then
  printf '%s\n' 'This script takes no arguments; edit its settings instead.' >&2
  exit 2
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The method illustration needs no training data or checkpoints.
python codes/evaluate/plot_transport_paths.py \
  --output_dir=outputs/visualizations/concept \
  --dpi=300
