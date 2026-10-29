#!/usr/bin/env bash

set -euo pipefail

# Resolve paths from the repository root.
cd "$(dirname "$0")/.."

PYTHON_BIN="${DAY7_PYTHON_BIN:-.venv/bin/python}"
PROFILE_OUTPUT_DIR="${DAY7_PROFILE_OUTPUT_DIR:-artifacts/results/day7/staging/profiling/raw}"
TRACE_DIR="${DAY7_TRACE_DIR:-artifacts/results/day7/local_archive/traces}"
OPTIMIZED_MODEL_DIR="${DAY7_OPTIMIZED_MODEL_DIR:-artifacts/results/day7/local_archive/optimized_models}"
DAY7_NOTES="${DAY7_NOTES:-}"

# 2 PyTorch profiles + 4 graph levels × 2 ORT thread settings.
EXPECTED_RESULT_COUNT=10
EXPECTED_TRACE_COUNT=10
EXPECTED_OPTIMIZED_MODEL_COUNT=4
RUN_COUNT=0

mkdir -p \
    "$PROFILE_OUTPUT_DIR" \
    "$TRACE_DIR" \
    "$OPTIMIZED_MODEL_DIR"

assert_empty_pattern() {
    local directory="$1"
    local pattern="$2"
    local label="$3"

    if find "$directory" \
        -maxdepth 1 \
        -type f \
        -name "$pattern" \
        -print -quit |
        grep -q .
    then
        printf '%s directory already contains matching files: %s\n' \
            "$label" \
            "$directory" >&2
        exit 1
    fi
}

# Do not mix compact results, traces, or optimized graphs from campaigns.
assert_empty_pattern \
    "$PROFILE_OUTPUT_DIR" \
    '*.json' \
    'Profile result'

assert_empty_pattern \
    "$TRACE_DIR" \
    '*.json' \
    'Trace'

assert_empty_pattern \
    "$OPTIMIZED_MODEL_DIR" \
    '*.onnx' \
    'Optimized-model'

# Keep thread policy controlled only by the explicit CLI arguments.
for variable in \
    OMP_NUM_THREADS \
    MKL_NUM_THREADS \
    OPENBLAS_NUM_THREADS \
    VECLIB_MAXIMUM_THREADS \
    NUMEXPR_NUM_THREADS
do
    if printenv "$variable" >/dev/null 2>&1; then
        value="$(printenv "$variable")"
        printf '%s must be unset for the profiling campaign; current value: %s\n' \
            "$variable" "${value:-<empty>}" >&2
        exit 1
    fi
done

# Pin all ten profiling processes to one committed source state.
CAMPAIGN_COMMIT="$(git rev-parse HEAD)"

assert_repository_state() {
    local current_commit
    local current_status

    current_commit="$(git rev-parse HEAD)"
    current_status="$(git status --porcelain)"

    if [[ "$current_commit" != "$CAMPAIGN_COMMIT" ]]; then
        printf 'Git HEAD changed during the campaign.\n' >&2
        printf 'Expected: %s\n' "$CAMPAIGN_COMMIT" >&2
        printf 'Current:  %s\n' "$current_commit" >&2
        exit 1
    fi

    if [[ -n "$current_status" ]]; then
        printf 'Working tree is not clean:\n%s\n' \
            "$current_status" >&2
        exit 1
    fi
}

run_profile() {
    local backend="$1"
    local threads="$2"
    local level="$3"
    local optimized_model_path="${4:-}"

    local command=(
        "$PYTHON_BIN"
        scripts/day7_profile_cpu.py
        --backend "$backend"
        --intra-op-threads "$threads"
        --graph-optimization-level "$level"
        --warmup 10
        --active-runs 10
        --output-dir "$PROFILE_OUTPUT_DIR"
        --trace-dir "$TRACE_DIR"
        --notes "$DAY7_NOTES"
    )

    # Serialize one representative optimized graph for each level during
    # the t1 profile. The t6 session applies the same optimization level
    # independently in memory without serializing another graph.
    if [[ -n "$optimized_model_path" ]]; then
        command+=(
            --optimized-model-path "$optimized_model_path"
        )
    fi

    assert_repository_state

    printf 'Running %d/%d: backend=%s graph_opt=%s intra_op=%s\n' \
        "$((RUN_COUNT + 1))" \
        "$EXPECTED_RESULT_COUNT" \
        "$backend" \
        "$level" \
        "$threads"

    # Every profile receives a fresh Python process and profiler session.
    "${command[@]}"

    RUN_COUNT=$((RUN_COUNT + 1))
}

# Profiling also requires a clean committed implementation.
assert_repository_state

run_profile \
    pytorch_cpu \
    1 \
    all

run_profile \
    pytorch_cpu \
    6 \
    all

for level in \
    disable \
    basic \
    extended \
    all
do
    run_profile \
        ort_cpu \
        1 \
        "$level" \
        "$OPTIMIZED_MODEL_DIR/$level.onnx"

    run_profile \
        ort_cpu \
        6 \
        "$level"
done

if [[ "$RUN_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]]; then
    printf 'Unexpected process count: expected=%d actual=%d\n' \
        "$EXPECTED_RESULT_COUNT" \
        "$RUN_COUNT" >&2
    exit 1
fi

ACTUAL_RESULT_COUNT="$(
    find "$PROFILE_OUTPUT_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.json' |
        wc -l |
        tr -d ' '
)"

ACTUAL_TRACE_COUNT="$(
    find "$TRACE_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.json' |
        wc -l |
        tr -d ' '
)"

ACTUAL_OPTIMIZED_MODEL_COUNT="$(
    find "$OPTIMIZED_MODEL_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.onnx' |
        wc -l |
        tr -d ' '
)"

if [[ "$ACTUAL_RESULT_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]] \
    || [[ "$ACTUAL_TRACE_COUNT" -ne "$EXPECTED_TRACE_COUNT" ]] \
    || [[ "$ACTUAL_OPTIMIZED_MODEL_COUNT" \
        -ne "$EXPECTED_OPTIMIZED_MODEL_COUNT" ]]
then
    printf 'Profiling artifact count mismatch: results=%d traces=%d optimized_models=%d\n' \
        "$ACTUAL_RESULT_COUNT" \
        "$ACTUAL_TRACE_COUNT" \
        "$ACTUAL_OPTIMIZED_MODEL_COUNT" >&2
    exit 1
fi

printf 'Profiling campaign completed: %d results, %d traces, %d optimized models.\n' \
    "$ACTUAL_RESULT_COUNT" \
    "$ACTUAL_TRACE_COUNT" \
    "$ACTUAL_OPTIMIZED_MODEL_COUNT"