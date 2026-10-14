#!/usr/bin/env bash

# Stop on failed commands, reject unset variables, and propagate pipeline
# failures. Each Python invocation below must complete successfully.
set -euo pipefail

# Resolve project-relative paths from the directory containing this script.
cd "$(dirname "$0")/.."

# Activate the project virtual environment before running this script.
# The child process inherit its PATH and therefore use its Python interpreter.
#
# Each invocation starts a fresh process with one selected bechmark backend.
# Rotate backend order so that every backend occupies each position once.
#
# Disable explicit MPS CPU fallback for each invocation. Forward optional
# experiment notes, using an empty string when DAY%_NOTES is unset or empty.

for backend in pytorch_cpu ort_cpu pytorch_mps
do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/day5_benchmark.py \
    --backend "$backend" --round 1 --notes "${DAY5_NOTES:-}"
done

for backend in ort_cpu pytorch_mps pytorch_cpu
do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/day5_benchmark.py \
    --backend "$backend" --round 2 --notes "${DAY5_NOTES:-}"
done

for backend in pytorch_mps pytorch_cpu ort_cpu
do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/day5_benchmark.py \
    --backend "$backend" --round 3 --notes "${DAY5_NOTES:-}"
done
