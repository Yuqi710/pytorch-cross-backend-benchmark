#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${DAY6_PYTHON_BIN:-python}"
OUTPUT_DIR="${DAY6_SMOKE_OUTPUT_DIR:-artifacts/results/day6/local_smoke/cpu_threads/raw}"
EXPECTED_RESULT_COUNT=45
RUN_COUNT=0

mkdir -p "$OUTPUT_DIR"

if find "$OUTPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.json' \
    -print -quit |
    grep -q .
then
    printf 'Smoke output directory already contains JSON results: %s\n' \
        "$OUTPUT_DIR" >&2
    exit 1
fi

run_case() {
    local backend="$1"
    local intra_op="$2"
    local round="$3"

    local command=(
        "$PYTHON_BIN" scripts/day6_cpu_threads.py
        --backend "$backend"
        --round "$round"
        --warmup 2
        --iterations 1
        --output-dir "$OUTPUT_DIR"
    )

    if [[ "$intra_op" != "default" ]]; then
        command+=(
            --intra-op-threads "$intra_op"
        )
    fi

    printf 'round=%s backend=%s intra_op=%s\n' \
        "$round" \
        "$backend" \
        "$intra_op"

    PYTORCH_ENABLE_MPS_FALLBACK=0 "${command[@]}"
    RUN_COUNT=$((RUN_COUNT + 1))
}

run_round() {
    local round="$1"

    local threads=()
    local backends=()

    case "$round" in
        1)
            threads=(default 1 2 4 6)
            backends=(pytorch_cpu ort_cpu pytorch_mps)
            ;;
        2)
            threads=(2 4 6 default 1)
            backends=(ort_cpu pytorch_mps pytorch_cpu)
            ;;
        3)
            threads=(6 default 1 2 4)
            backends=(pytorch_mps pytorch_cpu ort_cpu)
            ;;
    esac

    for intra_op in "${threads[@]}"
    do
        for backend in "${backends[@]}"
        do
            run_case "$backend" "$intra_op" "$round"
        done
    done
}

for round in 1 2 3
do
    run_round "$round"
done

ACTUAL_RESULT_COUNT="$(
    find "$OUTPUT_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.json' |
        wc -l |
        tr -d ' '
)"

if [[ "$RUN_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]] \
    || [[ "$ACTUAL_RESULT_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]]
then
    printf 'Smoke result count mismatch: runs=%d files=%d expected=%d\n' \
        "$RUN_COUNT" "$ACTUAL_RESULT_COUNT" "$EXPECTED_RESULT_COUNT" >&2
    exit 1
fi

printf 'Smoke campaign completed: %d JSON files.\n' \
    "$ACTUAL_RESULT_COUNT"
