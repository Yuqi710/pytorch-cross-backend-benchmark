"""Validate and summarize the formal Day 6 CPU intra-op thread sweep.

The module treats each process as an independent experimental run, verifies
the completeness and reproducibility of the full campaign, and derives
cross-process statistics without pooling raw latency samples across runs.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from inference_bench.result_validation import (
    field,
    finite_number,
    validate_ac_power,
    validate_correctness,
    validate_statistics,
    validate_thread_environment,
)


# Define the complete formal experiment matrix.
BACKENDS = ("pytorch_cpu", "ort_cpu", "pytorch_mps")
CPU_BACKENDS = ("pytorch_cpu", "ort_cpu")
THREAD_VALUES = (None, 1, 2, 4, 6)
ROUNDS = (1, 2, 3)

# Keep explicit thread values separate from the runtime-default policy.
EXPLICIT_THREAD_VALUES = (1, 2, 4, 6)

# Retain the per-process metrics needed for analysis and reporting.
STATISTIC_FIELDS = (
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "cv_percent",
)


def _thread_label(value: int | None) -> str:
    """Return the stable textual label for one thread configuration."""

    return "default" if value is None else str(value)


def _requested_threads(record: Mapping[str, Any]) -> int | None:
    """Validate and return the requested intra-op thread configuration."""

    policy = field(record, "threading.intra_op.policy")
    requested = field(
        record,
        "threading.intra_op.requested_threads",
    )

    if policy == "runtime_default" and requested is None:
        return None

    if (
        policy == "explicit"
        and type(requested) is int
        and requested in EXPLICIT_THREAD_VALUES
    ):
        return requested

    raise ValueError(
        "Intra-op configuration must be runtime_default or "
        "an explicit value in {1, 2, 4, 6}"
    )


def _fingerprint(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the fields that must remain identical across the campaign."""

    return {
        "git_commit": field(record, "source.git_commit"),
        "source_sha256": field(record, "source.source_sha256"),
        "dependencies": field(record, "environment.versions"),
        "python": field(record, "environment.python"),
        "architecture": field(record, "environment.architecture"),
        "system": field(record, "environment.system"),
        "macos_version": field(record, "environment.macos_version"),
        "logical_cpu_count": field(
            record,
            "environment.logical_cpu_count",
        ),
        "cpu_model": field(record, "environment.cpu_model"),
        "hardware_model": field(record, "environment.hardware_model"),
        "memory_bytes": field(record, "environment.memory_bytes"),
        "model_name": field(record, "model.name"),
        "model_weights": field(record, "model.weights"),
        "weights_sha256": field(record, "model.weights_sha256"),
        "input_sha256": field(record, "model.input_sha256"),
        "input_shape": field(record, "model.input_shape"),
        "precision": field(record, "model.precision"),
        "model_seed": field(record, "model.model_seed"),
        "input_seed": field(record, "model.input_seed"),
    }


def _validate_threading(
    record: Mapping[str, Any],
    backend: str,
    requested: int | None,
) -> None:
    """Validate intra-op control and unchanged inter-op policy."""

    if field(record, "threading.experiment_variable") != "intra_op_threads":
        raise ValueError("Invalid thread experiment variable")

    expected_role = (
        "mps_host_thread_sensitivity_control"
        if backend == "pytorch_mps"
        else "primary_cpu_scan"
    )
    if field(record, "threading.backend_role") != expected_role:
        raise ValueError(f"{backend}: invalid backend role")

    expected_scope = {
        "pytorch_cpu": "pytorch_cpu_operator_intra_op",
        "ort_cpu": "onnxruntime_cpu_operator_intra_op",
        "pytorch_mps": (
            "pytorch_cpu_intra_op_setting_not_mps_worker_count"
        ),
    }[backend]
    if field(record, "threading.intra_op.scope") != expected_scope:
        raise ValueError(f"{backend}: invalid intra-op scope")

    reported_intra = field(
        record,
        "threading.intra_op.runtime_reported_value",
    )

    # Verify that every explicit intra-op request took effect.
    if requested is not None and reported_intra != requested:
        raise ValueError(
            f"{backend}: requested intra-op {requested}, "
            f"runtime reported {reported_intra}"
        )

    # ORT uses zero as the sentinel for its runtime-default policy.
    if backend == "ort_cpu" and requested is None:
        if reported_intra != 0:
            raise ValueError(
                "ORT default intra-op configuration must report 0"
            )
    # PyTorch reports a positive configured CPU intra-op value.
    elif (
        type(reported_intra) is not int
        or reported_intra < 1
    ):
        raise ValueError(
            f"{backend}: invalid reported intra-op value"
        )

    # Keep inter-op at the runtime default for every experiment case.
    if (
        field(record, "threading.inter_op.policy")
        != "runtime_default"
    ):
        raise ValueError(
            f"{backend}: inter-op policy must remain runtime_default"
        )

    if (
        field(record, "threading.inter_op.requested_threads")
        is not None
    ):
        raise ValueError(
            f"{backend}: inter-op threads must not be explicitly set"
        )

    reported_inter = field(
        record,
        "threading.inter_op.runtime_reported_value",
    )

    if backend == "ort_cpu":
        if reported_inter != 0:
            raise ValueError(
                "ORT default inter-op configuration must report 0"
            )
    elif type(reported_inter) is not int or reported_inter < 1:
        raise ValueError(
            f"{backend}: invalid reported inter-op value"
        )

    if backend == "ort_cpu":
        native_intra = field(record, "backend.intra_op_num_threads")
        native_inter = field(record, "backend.inter_op_num_threads")
        if field(record, "backend.providers") != ["CPUExecutionProvider"]:
            raise ValueError("ORT must use only CPUExecutionProvider")
        if (
            field(record, "backend.execution_mode")
            != "ExecutionMode.ORT_SEQUENTIAL"
        ):
            raise ValueError("ORT execution mode must remain sequential")
        if (
            field(record, "backend.graph_optimization_level")
            != "GraphOptimizationLevel.ORT_ENABLE_ALL"
        ):
            raise ValueError("ORT graph optimization policy changed")
    else:
        native_intra = field(record, "backend.torch_num_threads")
        native_inter = field(record, "backend.torch_num_interop_threads")
        expected_device = "mps" if backend == "pytorch_mps" else "cpu"
        if field(record, "backend.device") != expected_device:
            raise ValueError(f"{backend}: invalid device")

    if native_intra != reported_intra:
        raise ValueError(
            f"{backend}: native intra-op value does not match normalized metadata"
        )
    if native_inter != reported_inter:
        raise ValueError(
            f"{backend}: native inter-op value does not match normalized metadata"
        )


def _validate_record(
    record: Mapping[str, Any],
) -> tuple[
    tuple[str, int | None, int],
    dict[str, Any],
]:
    """Validate one independent process result and return its indexed summary."""

    if field(record, "schema_version") != 1:
        raise ValueError("Formal schema version must be 1")
    if field(record, "experiment") != "day6_cpu_intra_op_thread_sweep":
        raise ValueError("Formal experiment name is invalid")
    if field(record, "status") != "passed":
        raise ValueError("Formal result status must be passed")

    benchmark = field(record, "benchmark")
    if benchmark != {"warmup": 50, "iterations": 300}:
        raise ValueError("Formal benchmark configuration must be 50/300")

    expected_measurement_scope = {
        "name": "prepared_input_synchronized_call",
        "timed_operations": ["backend.run", "backend.synchronize"],
        "request_concurrency": 1,
        "thread_policy": "intra_op_swept_inter_op_runtime_default",
    }
    for key, expected in expected_measurement_scope.items():
        if field(record, f"measurement_scope.{key}") != expected:
            raise ValueError("Formal measurement scope is invalid")

    validate_thread_environment(record)
    validate_ac_power(record)

    backend = field(record, "backend.name")
    round_number = field(record, "round")
    requested = _requested_threads(record)

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")

    if round_number not in ROUNDS:
        raise ValueError(f"Invalid round: {round_number}")

    # Require a clean source state for every formal experiment process.
    git_status = field(record, "source.git_status")
    if not isinstance(git_status, str) or git_status.strip():
        raise ValueError(
            f"{backend} round {round_number}: Git status is not clean"
        )

    # Require the complete fixed-size sequence of raw latency observation.
    samples = field(record, "samples_ns")
    if (
        not isinstance(samples, list)
        or len(samples) != 300
        or any(
            type(sample) is not int or sample <= 0
            for sample in samples
        )
    ):
        raise ValueError(
            f"{backend} round {round_number}: "
            "expected 300 positive integer samples"
        )

    validate_correctness(record)
    _validate_threading(record, backend, requested)

    # Require explicit MPS CPU fallback to remain disabled.
    if backend == "pytorch_mps":
        fallback = field(record, "backend.mps_fallback_env")
        if fallback != "0":
            raise ValueError(
                "MPS records require PYTORCH_ENABLE_MPS_FALLBACK=0"
            )

    wall_time = field(
        record,
        "cpu_process_observation.sampling_block_wall_time_ns",
    )
    process_cpu_time = field(
        record,
        "cpu_process_observation.sampling_block_process_cpu_time_ns",
    )
    if type(wall_time) is not int or wall_time <= 0:
        raise ValueError("Sampling block wall time must be positive")
    if type(process_cpu_time) is not int or process_cpu_time < 0:
        raise ValueError("Sampling block process CPU time is invalid")

    cpu_equivalents = finite_number(
        field(
            record,
            "cpu_process_observation."
            "average_process_cpu_core_equivalents",
        ),
        "average_process_cpu_core_equivalents",
    )

    if not math.isclose(
        cpu_equivalents,
        process_cpu_time / wall_time,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "CPU core equivalents do not match block times"
        )

    computed = validate_statistics(record, samples)

    # Reduce the validated record to metrics required by later aggregation.
    run_summary = {
        "round": round_number,
        "run_id": field(record, "run_id"),
        **{
            field: computed[field]
            for field in STATISTIC_FIELDS
        },
        "average_process_cpu_core_equivalents": cpu_equivalents,
    }

    # Use backend, requested intra-op value, and round as the unique case key.
    case_key = (
        backend,
        requested,
        round_number,
    )

    return case_key, run_summary


def _case_sort_key(
    key: tuple[str, int | None, int],
) -> tuple[int, int, int]:
    """Return a deterministic sort key for campaign case identifiers."""

    backend, threads, round_number = key

    return (
        BACKENDS.index(backend),
        THREAD_VALUES.index(threads),
        round_number,
    )


def validate_and_summarize_thread_sweep(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate all 45 records and produce the formal campaign summary."""

    # Reject incomplete campaigns before performing any aggregation.
    expected_record_count = (
        len(BACKENDS)
        * len(THREAD_VALUES)
        * len(ROUNDS)
    )
    if len(records) != expected_record_count:
        raise ValueError(
            f"Expected exactly {expected_record_count} records, "
            f"received {len(records)}"
        )

    reference_fingerprint = _fingerprint(records[0])
    indexed_runs: dict[
        tuple[str, int | None, int],
        dict[str, Any],
    ] = {}

    for record in records:
        if _fingerprint(record) != reference_fingerprint:
            raise ValueError(
                "Model, input, source, or dependency fingerprint differs"
            )

        key, run_summary = _validate_record(record)

        # Reject duplicate backend/thread/round combinations.
        if key in indexed_runs:
            raise ValueError(f"Duplicate campaign case: {key}")

        # Index each run by case key.
        indexed_runs[key] = run_summary

    expected_cases = {
        (backend, threads, round_number)
        for backend in BACKENDS
        for threads in THREAD_VALUES
        for round_number in ROUNDS
    }

    actual_cases = set(indexed_runs)

    # Compare the observed case set with the complete expected matrix.
    if actual_cases != expected_cases:
        missing = sorted(
            expected_cases - actual_cases,
            key=_case_sort_key,
        )
        unexpected = sorted(
            actual_cases - expected_cases,
            key=_case_sort_key,
        )
        raise ValueError(
            f"Incomplete campaign: missing={missing}, "
            f"unexpected={unexpected}"
        )

    groups: list[dict[str, Any]] = []
    group_index: dict[
        tuple[str, int | None],
        dict[str, Any],
    ] = {}

    # Aggregate three independent processes for each backend/thread group.
    for backend in BACKENDS:
        for threads in THREAD_VALUES:
            runs = [
                indexed_runs[(backend, threads, round_number)]
                for round_number in ROUNDS
            ]
            run_means = [run["mean_ms"] for run in runs]
            core_equivalents = [
                run["average_process_cpu_core_equivalents"]
                for run in runs
            ]

            group = {
                "backend": backend,
                "intra_op_label": _thread_label(threads),
                "requested_threads": threads,
                "runs": runs,
                # Use the median process mean as the representation latency.
                "representative_mean_ms": statistics.median(
                    run_means
                ),
                # Measure process-to-process range relative to process means.
                "cross_process_relative_range_percent": (
                    (max(run_means) - min(run_means))
                    / statistics.fmean(run_means)
                    * 100
                ),
                "representative_process_cpu_core_equivalents": (
                    statistics.median(core_equivalents)
                ),
            }

            groups.append(group)
            # Maintain direct lookup access for cross-group comparisons.
            group_index[(backend, threads)] = group

    # Express CPU configurations relative to explicit single-thread execution.
    for backend in CPU_BACKENDS:
        single_thread_mean = group_index[
            (backend, 1)
        ]["representative_mean_ms"]

        for threads in THREAD_VALUES:
            group = group_index[(backend, threads)]
            group["speedup_vs_explicit_1_thread_x"] = (
                single_thread_mean
                / group["representative_mean_ms"]
            )

    matched_cpu_configurations = []

    # Compare CPU runtimes only at matched explicit intra-op values.
    for threads in EXPLICIT_THREAD_VALUES:
        pytorch_mean = group_index[
            ("pytorch_cpu", threads)
        ]["representative_mean_ms"]
        ort_mean = group_index[
            ("ort_cpu", threads)
        ]["representative_mean_ms"]

        matched_cpu_configurations.append(
            {
                "requested_threads": threads,
                "pytorch_cpu_representative_mean_ms": pytorch_mean,
                "ort_cpu_representative_mean_ms": ort_mean,
                "pytorch_cpu_over_ort_cpu_latency_ratio": (
                    pytorch_mean / ort_mean
                ),
            }
        )

    default_vs_best_explicit = []

    # Identify the best explicit setting for each CPU backend.
    for backend in CPU_BACKENDS:
        default_group = group_index[(backend, None)]
        best_explicit = min(
            (
                group_index[(backend, threads)]
                for threads in EXPLICIT_THREAD_VALUES
            ),
            key=lambda group: group["representative_mean_ms"],
        )

        ratio = (
            default_group["representative_mean_ms"]
            / best_explicit["representative_mean_ms"]
        )

        default_vs_best_explicit.append(
            {
                "backend": backend,
                "default_representative_mean_ms": (
                    default_group["representative_mean_ms"]
                ),
                "best_explicit_threads": (
                    best_explicit["requested_threads"]
                ),
                "best_explicit_representative_mean_ms": (
                    best_explicit["representative_mean_ms"]
                ),
                "default_over_best_explicit_latency_ratio": ratio,
                "default_minus_best_explicit_percent": (
                    (ratio - 1) * 100
                ),
            }
        )

    mps_groups = [
        group_index[("pytorch_mps", threads)]
        for threads in THREAD_VALUES
    ]
    fastest_mps = min(
        mps_groups,
        key=lambda group: group["representative_mean_ms"],
    )
    slowest_mps = max(
        mps_groups,
        key=lambda group: group["representative_mean_ms"],
    )
    mps_means = [
        group["representative_mean_ms"]
        for group in mps_groups
    ]

    # Summarize sensitivity without interpreting settings as MPS thread counts.
    mps_host_thread_sensitivity = {
        "interpretation": (
            "Sensitivity to the PyTorch CPU intra-op setting; "
            "not an MPS worker-thread count."
        ),
        "observed_fastest_setting_label": (
            fastest_mps["intra_op_label"]
        ),
        "observed_slowest_setting_label": (
            slowest_mps["intra_op_label"]
        ),
        "representative_mean_range_ms": [
            min(mps_means),
            max(mps_means),
        ],
        "relative_range_percent": (
            (
                max(mps_means)
                - min(mps_means)
            )
            / statistics.fmean(mps_means)
            * 100
        ),
    }

    return {
        "schema_version": 1,
        "experiment": "day6_cpu_intra_op_thread_sweep_summary",
        "validation": {
            "record_count": len(records),
            "group_count": len(groups),
            "rounds_per_group": len(ROUNDS),
            "samples_per_run": 300,
            "status": "passed",
        },
        "aggregation": {
            "representative_latency": "median_of_process_means",
            "cross_process_range": (
                "(max(run_means)-min(run_means))/"
                "mean(run_means)*100"
            ),
            "raw_sample_pooling": False,
        },
        "fingerprint": reference_fingerprint,
        "groups": groups,
        "matched_cpu_configurations": matched_cpu_configurations,
        "default_vs_best_explicit": default_vs_best_explicit,
        "mps_host_thread_sensitivity": (
            mps_host_thread_sensitivity
        ),
    }


def thread_sweep_csv_rows(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten the 15 configuration groups into CSV rows."""

    rows: list[dict[str, Any]] = []

    for group in summary["groups"]:
        # Store one row per backend/thread configuration.
        row = {
            "backend": group["backend"],
            "intra_op_label": group["intra_op_label"],
            "requested_threads": (
                ""
                if group["requested_threads"] is None
                else group["requested_threads"]
            ),
            "representative_mean_ms": (
                group["representative_mean_ms"]
            ),
            "cross_process_relative_range_percent": (
                group[
                    "cross_process_relative_range_percent"
                ]
            ),
            "representative_process_cpu_core_equivalents": (
                group[
                    "representative_process_cpu_core_equivalents"
                ]
            ),
            "speedup_vs_explicit_1_thread_x": (
                group.get("speedup_vs_explicit_1_thread_x")
            ),
        }

        # Expand each independent round into deterministic CSV columns.
        for run in group["runs"]:
            prefix = f"round_{run['round']}"

            for field in STATISTIC_FIELDS:
                row[f"{prefix}_{field}"] = run[field]

            row[
                f"{prefix}_average_process_cpu_core_equivalents"
            ] = run[
                "average_process_cpu_core_equivalents"
            ]

        rows.append(row)

    return rows
