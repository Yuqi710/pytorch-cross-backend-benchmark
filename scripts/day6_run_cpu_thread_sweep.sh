#!/usr/bin/env bash

set -euo pipefail

# Resolve all project-relative paths from the repository root, regardless of
# the caller's current working directory.
cd "$(dirname "$0")/.."

# Allow the caller to select a Python interpreter, result directory, and
# campaign-wide notes. Use reproducible project defaults when they are unset.
PYTHON_BIN="${DAY6_PYTHON_BIN:-python}"
DAY6_OUTPUT_DIR="${DAY6_OUTPUT_DIR:-artifacts/results/day6/staging/cpu_threads/raw}"
DAY6_NOTES="${DAY6_NOTES:-}"

# The full matrix contains three rounds, five intra-op configurations, and
# three backends: 3 × 5 × 3 = 45 independent processes.
EXPECTED_RESULT_COUNT=45
RUN_COUNT=0

# Collect formal results in an ignored staging directory. Require an empty
# directory so that files from separate campaigns cannot be mixed.
mkdir -p "$DAY6_OUTPUT_DIR"

if find "$DAY6_OUTPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.json' \
    -print -quit |
    grep -q .
then
    printf 'Output directory already contains JSON results: %s\n' \
        "$DAY6_OUTPUT_DIR" >&2
    exit 1
fi

# Reject external thread-count overrides that could introduce additional
# experimental variables beyond the runtime configurations under test.
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

# Pin the campaign to the commit that is checked out when collection begins.
CAMPAIGN_COMMIT="$(git rev-parse HEAD)"

assert_repository_state() {
    local current_commit
    local current_status

    current_commit="$(git rev-parse HEAD)"
    current_status="$(git status --porcelain)"

    # All result files in one campaign must refer to the same Git commit.
    if [[ "$current_commit" != "$CAMPAIGN_COMMIT" ]]; then
        printf 'Git HEAD changed during the campaign.\n' >&2
        printf 'Expected: %s\n' "$CAMPAIGN_COMMIT" >&2
        printf 'Current:  %s\n' "$current_commit" >&2
        exit 1
    fi

    # Formal collection requires a clean working tree. The staging result
    # directory and generated model files must already be ignored by Git.
    if [[ -n "$current_status" ]]; then
        printf 'Working tree is not clean:\n%s\n' \
            "$current_status" >&2
        exit 1
    fi
}

run_case() {
    local backend="$1"
    local intra_op="$2"
    local round="$3"

    local command=(
        "$PYTHON_BIN" scripts/day6_cpu_threads.py
        --backend "$backend"
        --round "$round"
        --warmup 50
        --iterations 300
        --output-dir "$DAY6_OUTPUT_DIR"
        --notes "$DAY6_NOTES"
    )

    # Omitting --intra-op-threads selects the runtime's default policy.
    if [[ "$intra_op" != "default" ]]; then
        command+=(
            --intra-op-threads "$intra_op"
        )
    fi

    # Verify repository identity immediately before every independent run.
    assert_repository_state

    printf 'Running %d/%d: round=%s backend=%s intra_op=%s\n' \
        "$((RUN_COUNT + 1))" \
        "$EXPECTED_RESULT_COUNT" \
        "$round" \
        "$backend" \
        "$intra_op"

    # Start one fresh Python process for exactly one backend, thread
    # configuration, and round. Explicit MPS CPU fallback remains disabled.
    PYTORCH_ENABLE_MPS_FALLBACK=0 "${command[@]}"

    # Increment the completed-run count only after the child process exits
    # successfully.
    RUN_COUNT=$((RUN_COUNT + 1))
}

run_round() {
    local round="$1"

    local threads=()
    local backends=()

    # Rotate both dimensions across rounds to reduce fixed execution-position
    # bias while preserving the complete backend-by-thread matrix in each round.
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
        *)
            printf 'Unsupported round: %s\n' "$round" >&2
            exit 1
            ;;
    esac

    # Use the thread configuration as the outer loop and the backend as the
    # inner loop. Each round therefore covers the complete 5 × 3 Cartesian
    # product exactly once.
    for intra_op in "${threads[@]}"
    do
        for backend in "${backends[@]}"
        do
            run_case "$backend" "$intra_op" "$round"
        done
    done
}

# Execute all cases sequentially so that benchmark processes do not compete
# with one another for CPU or MPS resources.
for round in 1 2 3
do
    run_round "$round"
done

# Verify that the matrix expansion completed the expected number of successful
# child processes.
if [[ "$RUN_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]]; then
    printf 'Unexpected process count: expected=%d actual=%d\n' \
        "$EXPECTED_RESULT_COUNT" "$RUN_COUNT" >&2
    exit 1
fi

# Count only JSON files written directly into the campaign output directory.
ACTUAL_RESULT_COUNT="$(
    find "$DAY6_OUTPUT_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.json' |
        wc -l |
        tr -d ' '
)"

# Each successful child process must produce exactly one result file.
if [[ "$ACTUAL_RESULT_COUNT" -ne "$EXPECTED_RESULT_COUNT" ]]; then
    printf 'Unexpected JSON count: expected=%d actual=%d\n' \
        "$EXPECTED_RESULT_COUNT" \
        "$ACTUAL_RESULT_COUNT" >&2
    exit 1
fi

printf 'Campaign completed: %d independent processes, %d JSON files.\n' \
    "$RUN_COUNT" \
    "$ACTUAL_RESULT_COUNT"
