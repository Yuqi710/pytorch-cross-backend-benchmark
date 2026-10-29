#!/usr/bin/env bash

set -euo pipefail

# Resolve paths from the repository root.
cd "$(dirname "$0")/.."

PYTHON_BIN="${DAY7_PYTHON_BIN:-.venv/bin/python}"
DAY7_OUTPUT_DIR="${DAY7_GRAPH_OUTPUT_DIR:-artifacts/results/day7/staging/graph_optimization/raw}"
DAY7_NOTES="${DAY7_NOTES:-}"

# 4 graph levels × 2 thread settings × 3 rounds.
EXPECTED_RESULT_COUNT=24
RUN_COUNT=0

mkdir -p "$DAY7_OUTPUT_DIR"

# Refuse to mix results from different campaigns.
if find "$DAY7_OUTPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.json' \
    -print -quit |
    grep -q .
then
    printf 'Output directory already contains JSON results: %s\n' \
        "$DAY7_OUTPUT_DIR" >&2
    exit 1
fi

# External thread-pool settings would introduce uncontrolled variables.
for variable in \
    OMP_NUM_THREADS \
    MKL_NUM_THREADS \
    OPENBLAS_NUM_THREADS \
    VECLIB_MAXIMUM_THREADS \
    NUMEXPR_NUM_THREADS
do
    if printenv "$variable" >/dev/null 2>&1; then
        value="$(printenv "$variable")"
        printf '%s must be unset for the formal campaign; current value: %s\n' \
            "$variable" "${value:-<empty>}" >&2
        exit 1
    fi
done

# Pin the complete campaign to one committed source state.
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

run_case() {
    local level="$1"
    local threads="$2"
    local round="$3"

    local command=(
        "$PYTHON_BIN"
        scripts/day7_graph_optimization.py
        --graph-optimization-level "$level"
        --intra-op-threads "$threads"
        --round "$round"
        --warmup 50
        --iterations 300
        --output-dir "$DAY7_OUTPUT_DIR"
        --notes "$DAY7_NOTES"
    )

    # Verify repository identity immediately before every independent run.
    assert_repository_state

    printf 'Running %d/%d: round=%s graph_opt=%s intra_op=%s\n' \
        "$((RUN_COUNT + 1))" \
        "$EXPECTED_RESULT_COUNT" \
        "$round" \
        "$level" \
        "$threads"

    # One process executes exactly one graph/thread/round configuration.
    "${command[@]}"

    RUN_COUNT=$((RUN_COUNT + 1))
}

run_round() {
    local round="$1"
    local cases=()

    # Every round covers the complete 4 × 2 matrix exactly once.
    # Configuration order changes across rounds to reduce fixed-position bias.
    case "$round" in
        1)
            cases=(
                disable:1
                basic:6
                extended:1
                all:6
                disable:6
                basic:1
                extended:6
                all:1
            )
            ;;
        2)
            cases=(
                extended:6
                all:1
                disable:6
                basic:1
                extended:1
                all:6
                disable:1
                basic:6
            )
            ;;
        3)
            cases=(
                all:6
                extended:1
                basic:6
                disable:1
                all:1
                extended:6
                basic:1
                disable:6
            )
            ;;
        *)
            printf 'Unsupported round: %s\n' \
                "$round" >&2
            exit 1
            ;;
    esac

    local case_value
    local level
    local threads

    for case_value in "${cases[@]}"
    do
        level="${case_value%%:*}"
        threads="${case_value##*:}"

        run_case \
            "$level" \
            "$threads" \
            "$round"
    done
}

# Formal collection starts only from a clean committed worktree.
assert_repository_state

for round in 1 2 3
do
    run_round "$round"
done

if [[ "$RUN_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]]; then
    printf 'Unexpected process count: expected=%d actual=%d\n' \
        "$EXPECTED_RESULT_COUNT" "$RUN_COUNT" >&2
    exit 1
fi

ACTUAL_RESULT_COUNT="$(
    find "$DAY7_OUTPUT_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.json' |
        wc -l |
        tr -d ' '
)"

if [[ "$ACTUAL_RESULT_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]]; then
    printf 'Unexpected JSON count: expected=%d actual=%d\n' \
        "$EXPECTED_RESULT_COUNT" \
        "$ACTUAL_RESULT_COUNT" >&2
    exit 1
fi

printf 'Graph-optimization campaign completed: %d processes, %d JSON files.\n' \
    "$RUN_COUNT" \
    "$ACTUAL_RESULT_COUNT"
