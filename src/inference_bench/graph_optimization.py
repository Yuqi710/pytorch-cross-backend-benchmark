"""Validate and summarize the formal Day 7 ORT graph-optimization campaign.

Each process remains an independent experimental run. The module validates the
complete graph-level/thread/round matrix, recomputes statistics from raw latency
samples, and aggregates process-level metrics without pooling samples across
processes.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping

from inference_bench.ort_backend import GRAPH_OPTIMIZATION_LEVELS
from inference_bench.result_validation import (
    field,
    finite_number,
    validate_ac_power,
    validate_correctness,
    validate_statistics,
    validate_thread_environment,
)


# The formal matrix contains four graph levels, two thread counts, and three
# independent process rounds: 24 records grouped into eight configurations.
GRAPH_LEVELS = tuple(GRAPH_OPTIMIZATION_LEVELS)
THREAD_COUNTS = (1, 6)
ROUNDS = (1, 2, 3)

STATISTIC_FIELDS = (
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "cv_percent",
)


def _fingerprint(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract fields that must remain identical across all 24 processes."""

    # Graph level, thread count, round, and the post-run power snapshot vary by
    # case. Source, host, workload, ONNX artifact, and measurement policy do not.
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
        "thread_environment": field(
            record,
            "environment.thread_environment",
        ),

        "measurement_scope": field(record, "measurement_scope"),
        "benchmark": field(record, "benchmark"),

        "model_name": field(record, "model.name"),
        "model_weights": field(record, "model.weights"),
        "weights_sha256": field(record, "model.weights_sha256"),
        "input_sha256": field(record, "model.input_sha256"),
        "input_shape": field(record, "model.input_shape"),
        "precision": field(record, "model.precision"),
        "model_seed": field(record, "model.model_seed"),
        "input_seed": field(record, "model.input_seed"),

        "onnx_path": field(record, "model.onnx.path"),
        "onnx_opset_version": field(
            record,
            "model.onnx.opset_version",
        ),
        "onnx_sha256": field(record, "model.onnx.sha256"),
        "onnx_file_size_bytes": field(
            record,
            "model.onnx.file_size_bytes",
        ),
    }


def _validate_environment(record: Mapping[str, Any]) -> None:
    """Validate the controlled formal experiment environment."""

    # External thread variables must remain unset and every run must finish on
    # AC power so the matrix compares the intended ORT session controls.
    validate_thread_environment(record)
    validate_ac_power(record)


def _validate_backend(
    record: Mapping[str, Any],
    level: str,
    threads: int,
) -> None:
    """Validate ORT graph, provider, execution, and threading metadata."""

    if field(record, "backend.name") != "ort_cpu":
        raise ValueError("Formal graph optimization requires ort_cpu")
    if field(record, "backend.runtime") != "onnxruntime":
        raise ValueError("Invalid backend runtime")
    if field(record, "backend.device") != "cpu":
        raise ValueError("ORT graph optimization requires the CPU device")
    if field(record, "backend.precision") != "fp32":
        raise ValueError("Formal graph optimization requires FP32")
    if field(record, "backend.providers") != ["CPUExecutionProvider"]:
        raise ValueError("ORT must use only CPUExecutionProvider")

    expected_runtime_level = str(
        GRAPH_OPTIMIZATION_LEVELS[level]
    )

    # The recorded symbolic level and ORT runtime enum must agree at every
    # metadata layer; the runtime enum is derived from the shared level map.
    graph_fields = (
        field(record, "graph_optimization.requested_level"),
        field(record, "graph_optimization.runtime_level_name"),
        field(record, "backend.graph_optimization_level_name"),
    )
    if graph_fields != (level, level, level):
        raise ValueError(
            f"Requested and runtime graph levels differ for {level}"
        )

    if (
        field(record, "graph_optimization.runtime_level")
        != expected_runtime_level
        or field(record, "backend.graph_optimization_level")
        != expected_runtime_level
    ):
        raise ValueError(
            f"ORT runtime graph level does not match {level}"
        )

    expected_execution_mode = "ExecutionMode.ORT_SEQUENTIAL"
    if (
        field(record, "graph_optimization.execution_mode")
        != expected_execution_mode
        or field(record, "backend.execution_mode")
        != expected_execution_mode
    ):
        raise ValueError("ORT execution mode must remain sequential")

    if (
        field(record, "threading.experiment_variable")
        != "intra_op_threads"
    ):
        raise ValueError("Invalid thread experiment variable")

    if field(record, "threading.intra_op.policy") != "explicit":
        raise ValueError("Intra-op thread policy must be explicit")
    if (
        field(record, "threading.intra_op.scope")
        != "onnxruntime_cpu_operator_intra_op"
    ):
        raise ValueError("Invalid ORT intra-op scope")

    if (
        field(record, "threading.intra_op.requested_threads")
        != threads
        or field(record, "threading.intra_op.runtime_reported_value")
        != threads
        or field(record, "backend.intra_op_num_threads")
        != threads
    ):
        raise ValueError(
            f"ORT intra-op metadata does not match t{threads}"
        )

    if (
        field(record, "threading.inter_op.policy")
        != "runtime_default"
        or field(record, "threading.inter_op.requested_threads")
        is not None
        or field(record, "threading.inter_op.runtime_reported_value")
        != 0
        or field(record, "backend.inter_op_num_threads") != 0
    ):
        raise ValueError("ORT inter-op policy must remain runtime default")

    if (
        field(record, "diagnostic_features.profiling_enabled")
        is not False
        or field(
            record,
            "diagnostic_features."
            "optimized_model_serialization_enabled",
        )
        is not False
    ):
        # Profiling and graph serialization add work that does not belong in
        # the formal latency measurement path.
        raise ValueError(
            "Formal latency sessions must disable diagnostic features"
        )


def _validate_record(
    record: Mapping[str, Any],
) -> tuple[tuple[str, int, int], dict[str, Any]]:
    """Validate one process result and return its case and metrics."""

    # Establish the formal measurement contract before consuming samples or
    # accepting the record as one matrix case.
    if field(record, "schema_version") != 1:
        raise ValueError("Day 7 schema version must be 1")
    if field(record, "experiment") != "day7_ort_graph_optimization":
        raise ValueError("Invalid Day 7 graph experiment name")
    if field(record, "status") != "passed":
        raise ValueError("Formal result status must be passed")
    if field(record, "benchmark") != {
        "warmup": 50,
        "iterations": 300,
    }:
        raise ValueError("Formal benchmark configuration must be 50/300")

    if (
        field(record, "measurement_scope.name")
        != "prepared_input_synchronized_call"
        or field(record, "measurement_scope.request_concurrency")
        != 1
        or field(record, "measurement_scope.thread_policy")
        != "explicit_intra_op_inter_op_runtime_default"
        or field(record, "measurement_scope.graph_optimization_policy")
        != "one_level_per_independent_process"
    ):
        raise ValueError("Invalid formal measurement scope")

    level = field(record, "graph_optimization.requested_level")
    threads = field(record, "threading.intra_op.requested_threads")
    round_number = field(record, "round")

    if level not in GRAPH_LEVELS:
        raise ValueError(f"Invalid graph optimization level: {level}")
    if type(threads) is not int or threads not in THREAD_COUNTS:
        raise ValueError(f"Invalid intra-op thread count: {threads}")
    if type(round_number) is not int or round_number not in ROUNDS:
        raise ValueError(f"Invalid round: {round_number}")

    run_id = field(record, "run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a nonempty string")

    for path in (
        "source.git_commit",
        "source.source_sha256",
        "model.weights_sha256",
        "model.input_sha256",
        "model.onnx.sha256",
    ):
        value = field(record, path)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path} must be a nonempty string")

    git_status = field(record, "source.git_status")
    if not isinstance(git_status, str) or git_status.strip():
        raise ValueError(
            f"{level}/t{threads}/r{round_number}: Git status is not clean"
        )

    samples = field(record, "samples_ns")
    # Raw per-iteration samples are the retained evidence for recomputing every
    # reported latency statistic; aggregate fields are never trusted alone.
    if (
        not isinstance(samples, list)
        or len(samples) != 300
        or any(
            type(sample) is not int or sample <= 0
            for sample in samples
        )
    ):
        raise ValueError(
            f"{level}/t{threads}/r{round_number}: "
            "expected 300 positive integer samples"
        )

    validate_correctness(record)
    _validate_environment(record)
    _validate_backend(record, level, threads)

    wall_time_ns = field(
        record,
        "cpu_process_observation.sampling_block_wall_time_ns",
    )
    process_cpu_time_ns = field(
        record,
        "cpu_process_observation."
        "sampling_block_process_cpu_time_ns",
    )

    if type(wall_time_ns) is not int or wall_time_ns <= 0:
        raise ValueError("Sampling block wall time must be positive")
    if (
        type(process_cpu_time_ns) is not int
        or process_cpu_time_ns < 0
    ):
        raise ValueError("Sampling block process CPU time is invalid")

    core_equivalents = finite_number(
        field(
            record,
            "cpu_process_observation."
            "average_process_cpu_core_equivalents",
        ),
        "average_process_cpu_core_equivalents",
    )
    # Process CPU time divided by wall time is the average number of logical
    # cores occupied during the measured sampling block.
    if not math.isclose(
        core_equivalents,
        process_cpu_time_ns / wall_time_ns,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Average CPU core equivalents do not match block times"
        )

    computed = validate_statistics(record, samples)

    run_summary = {
        "round": round_number,
        "run_id": run_id,
        **{
            name: computed[name]
            for name in STATISTIC_FIELDS
        },
        "average_process_cpu_core_equivalents": core_equivalents,
    }

    return (level, threads, round_number), run_summary


def validate_and_summarize_graph_optimization(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Validate all 24 records and produce the formal Day 7 summary."""

    expected_record_count = (
        len(GRAPH_LEVELS)
        * len(THREAD_COUNTS)
        * len(ROUNDS)
    )
    if len(records) != expected_record_count:
        raise ValueError(
            f"Expected exactly {expected_record_count} records, "
            f"received {len(records)}"
        )

    reference_fingerprint = _fingerprint(records[0])
    indexed_runs: dict[
        tuple[str, int, int],
        dict[str, Any],
    ] = {}
    run_ids: set[str] = set()

    for record in records:
        # Independent processes may differ only in the three declared matrix
        # coordinates; all experimental controls must match the reference.
        if _fingerprint(record) != reference_fingerprint:
            raise ValueError(
                "Source, environment, model, input, or ONNX "
                "fingerprint differs across the campaign"
            )

        case, run_summary = _validate_record(record)

        if case in indexed_runs:
            raise ValueError(f"Duplicate campaign case: {case}")

        run_id = run_summary["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate run_id: {run_id}")

        indexed_runs[case] = run_summary
        run_ids.add(run_id)

    expected_cases = {
        (level, threads, round_number)
        for level in GRAPH_LEVELS
        for threads in THREAD_COUNTS
        for round_number in ROUNDS
    }

    if set(indexed_runs) != expected_cases:
        missing = expected_cases - set(indexed_runs)
        unexpected = set(indexed_runs) - expected_cases
        raise ValueError(
            f"Incomplete campaign: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    groups: list[dict[str, Any]] = []
    group_index: dict[tuple[str, int], dict[str, Any]] = {}

    for level in GRAPH_LEVELS:
        for threads in THREAD_COUNTS:
            # Preserve each round as an independent process result. The
            # campaign summarizes process metrics instead of pooling samples.
            runs = [
                indexed_runs[(level, threads, round_number)]
                for round_number in ROUNDS
            ]

            process_mean_ms = [
                run["mean_ms"] for run in runs
            ]
            # The median process metric is the representative value for the
            # configuration and limits the influence of one unstable process.
            representative_mean_ms = statistics.median(
                process_mean_ms
            )
            min_mean_ms = min(process_mean_ms)
            max_mean_ms = max(process_mean_ms)

            group = {
                "graph_optimization_level": level,
                "intra_op_threads": threads,
                "process_count": len(runs),
                "runs": runs,
                "process_mean_ms": process_mean_ms,
                "representative_mean_ms": representative_mean_ms,
                "min_process_mean_ms": min_mean_ms,
                "max_process_mean_ms": max_mean_ms,
                "cross_process_range_percent": (
                    (max_mean_ms - min_mean_ms)
                    / representative_mean_ms
                    * 100
                ),
                "representative_p50_ms": statistics.median(
                    [run["p50_ms"] for run in runs]
                ),
                "representative_p90_ms": statistics.median(
                    [run["p90_ms"] for run in runs]
                ),
                "representative_p99_ms": statistics.median(
                    [run["p99_ms"] for run in runs]
                ),
                "representative_cv_percent": statistics.median(
                    [run["cv_percent"] for run in runs]
                ),
                "representative_core_equivalents": statistics.median(
                    [
                        run["average_process_cpu_core_equivalents"]
                        for run in runs
                    ]
                ),
            }

            groups.append(group)
            group_index[(level, threads)] = group

    for group in groups:
        threads = group["intra_op_threads"]
        # Graph-level speedups use disabled optimization at the same thread
        # count as their baseline, isolating graph policy from thread policy.
        baseline = group_index[("disable", threads)]
        baseline_mean = baseline["representative_mean_ms"]
        current_mean = group["representative_mean_ms"]

        group["speedup_vs_disable"] = (
            baseline_mean / current_mean
        )
        group["latency_change_vs_disable_percent"] = (
            (current_mean / baseline_mean - 1) * 100
        )

    thread_scaling = []

    for level in GRAPH_LEVELS:
        # Thread scaling compares t1 and t6 within the same graph level.
        t1_mean = group_index[(level, 1)]["representative_mean_ms"]
        t6_mean = group_index[(level, 6)]["representative_mean_ms"]

        thread_scaling.append(
            {
                "graph_optimization_level": level,
                "t1_representative_mean_ms": t1_mean,
                "t6_representative_mean_ms": t6_mean,
                "latency_ratio": t6_mean / t1_mean,
                "speedup": t1_mean / t6_mean,
            }
        )

    return {
        "schema_version": 1,
        "experiment": "day7_ort_graph_optimization_summary",
        "validation": {
            "record_count": len(records),
            "group_count": len(groups),
            "process_count_per_group": len(ROUNDS),
            "graph_optimization_levels": list(GRAPH_LEVELS),
            "intra_op_thread_counts": list(THREAD_COUNTS),
            "rounds": list(ROUNDS),
            "samples_per_process": 300,
            "status": "passed",
        },
        "campaign": {
            "fingerprint": reference_fingerprint,
        },
        "groups": groups,
        "thread_scaling_t1_to_t6": thread_scaling,
    }


def graph_optimization_csv_rows(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten the eight formal groups into deterministic CSV rows."""

    # Index the single t1-to-t6 comparison for each graph level so groups can
    # retrieve it without depending on the summary list order.
    scaling_by_level = {
        row["graph_optimization_level"]: row
        for row in summary["thread_scaling_t1_to_t6"]
    }

    rows = []

    for group in summary["groups"]:
        level = group["graph_optimization_level"]
        process_means = group["process_mean_ms"]
        scaling = scaling_by_level[level]

        # Repeat each level-wide thread-scaling comparison on both thread rows,
        # while retaining the three process means as separate round columns.
        rows.append(
            {
                "graph_optimization_level": level,
                "intra_op_threads": group["intra_op_threads"],
                "process_count": group["process_count"],
                "round_1_mean_ms": process_means[0],
                "round_2_mean_ms": process_means[1],
                "round_3_mean_ms": process_means[2],
                "representative_mean_ms": (
                    group["representative_mean_ms"]
                ),
                "min_process_mean_ms": (
                    group["min_process_mean_ms"]
                ),
                "max_process_mean_ms": (
                    group["max_process_mean_ms"]
                ),
                "cross_process_range_percent": (
                    group["cross_process_range_percent"]
                ),
                "representative_p50_ms": (
                    group["representative_p50_ms"]
                ),
                "representative_p90_ms": (
                    group["representative_p90_ms"]
                ),
                "representative_p99_ms": (
                    group["representative_p99_ms"]
                ),
                "representative_cv_percent": (
                    group["representative_cv_percent"]
                ),
                "representative_core_equivalents": (
                    group["representative_core_equivalents"]
                ),
                "speedup_vs_disable": (
                    group["speedup_vs_disable"]
                ),
                "latency_change_vs_disable_percent": (
                    group[
                        "latency_change_vs_disable_percent"
                    ]
                ),
                "t1_representative_mean_ms": (
                    scaling["t1_representative_mean_ms"]
                ),
                "t6_representative_mean_ms": (
                    scaling["t6_representative_mean_ms"]
                ),
                "t6_to_t1_latency_ratio": (
                    scaling["latency_ratio"]
                ),
                "t1_to_t6_speedup": scaling["speedup"],
            }
        )

    return rows
