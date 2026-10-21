"""Unit tests for Day6 thread-sweep validation and aggregation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from inference_bench.thread_sweep import (
    validate_and_summarize_thread_sweep,
)


BACKENDS = ("pytorch_cpu", "ort_cpu", "pytorch_mps")
INTRA_OP_VALUES = (None, 1, 2, 4, 6)
ROUNDS = (1, 2, 3)


def _synthetic_mean_ms(
    backend: str,
    intra_op_threads: int | None,
    round_number: int,
) -> float:
    """Return deterministic per-process means for aggregation tests."""

    # Use a distinctive series for one group so that the median and
    # cross-process range have simple independently calculated values.
    if backend == "pytorch_cpu" and intra_op_threads == 2:
        return {
            1: 10.0,
            2: 12.0,
            3: 14.0,
        }[round_number]

    backend_base = {
        "pytorch_cpu": 20.0,
        "ort_cpu": 4.0,
        "pytorch_mps": 8.0,
    }[backend]

    thread_offset = {
        None: 0.0,
        1: 5.0,
        2: 2.0,
        4: 1.0,
        6: 1.5,
    }[intra_op_threads]

    return backend_base + thread_offset + round_number / 10


def _thread_metadata(
    backend: str,
    intra_op_threads: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build normalized thread metadata and native backend metadata."""

    explicit = intra_op_threads is not None

    if backend == "ort_cpu":
        reported_intra_op = (
            intra_op_threads
            if explicit
            else 0
        )
        reported_inter_op = 0
        role = "primary_cpu_scan"
        scope = "onnxruntime_cpu_operator_intra_op"

        backend_metadata = {
            "name": backend,
            "runtime": "onnxruntime",
            "device": "cpu",
            "providers": ["CPUExecutionProvider"],
            "execution_mode": "ExecutionMode.ORT_SEQUENTIAL",
            "graph_optimization_level": (
                "GraphOptimizationLevel.ORT_ENABLE_ALL"
            ),
            "intra_op_num_threads": reported_intra_op,
            "inter_op_num_threads": reported_inter_op,
        }
    else:
        reported_intra_op = (
            intra_op_threads
            if explicit
            else 6
        )
        reported_inter_op = 18

        role = (
            "primary_cpu_scan"
            if backend == "pytorch_cpu"
            else "mps_host_thread_sensitivity_control"
        )
        scope = (
            "pytorch_cpu_operator_intra_op"
            if backend == "pytorch_cpu"
            else (
                "pytorch_cpu_intra_op_setting_"
                "not_mps_worker_count"
            )
        )

        backend_metadata = {
            "name": backend,
            "runtime": "pytorch",
            "device": (
                "cpu"
                if backend == "pytorch_cpu"
                else "mps"
            ),
            "torch_num_threads": reported_intra_op,
            "torch_num_interop_threads": reported_inter_op,
            "mps_fallback_env": "0",
        }

    threading = {
        "experiment_variable": "intra_op_threads",
        "backend_role": role,
        "intra_op": {
            "policy": (
                "explicit"
                if explicit
                else "runtime_default"
            ),
            "requested_threads": intra_op_threads,
            "runtime_reported_value": reported_intra_op,
            "scope": scope,
        },
        "inter_op": {
            "policy": "runtime_default",
            "requested_threads": None,
            "runtime_reported_value": reported_inter_op,
        },
    }

    return threading, backend_metadata


def _make_record(
    backend: str,
    intra_op_threads: int | None,
    round_number: int,
) -> dict[str, Any]:
    """Create one valid synthetic Day6 process result."""

    threading, backend_metadata = _thread_metadata(
        backend,
        intra_op_threads,
    )
    requested_mean_ms = _synthetic_mean_ms(
        backend,
        intra_op_threads,
        round_number,
    )
    sample_ns = round(requested_mean_ms * 1_000_000)
    mean_ms = sample_ns / 1_000_000

    thread_label = (
        "default"
        if intra_op_threads is None
        else str(intra_op_threads)
    )

    return {
        "schema_version": 1,
        "experiment": "day6_cpu_intra_op_thread_sweep",
        "run_id": (
            f"{backend}-t{thread_label}-r{round_number}"
        ),
        "timestamp_utc": (
            f"2026-10-20T00:00:0{round_number}+00:00"
        ),
        "round": round_number,
        "status": "passed",
        "source": {
            "git_commit": "implementation-commit",
            "git_status": "",
            "source_sha256": "shared-source-hash",
        },
        "environment": {
            "python": "3.11.14",
            "hardware_model": "synthetic-hardware",
            "architecture": "arm64",
            "system": "Darwin",
            "macos_version": "26.0.1",
            "logical_cpu_count": 12,
            "cpu_model": "synthetic-cpu",
            "memory_bytes": "34359738368",
            "power_source_snapshot_after_run": (
                "Now drawing from 'AC Power'"
            ),
            "thread_environment": {
                "OMP_NUM_THREADS": None,
                "MKL_NUM_THREADS": None,
                "OPENBLAS_NUM_THREADS": None,
                "VECLIB_MAXIMUM_THREADS": None,
                "NUMEXPR_NUM_THREADS": None,
            },
            "versions": {
                "torch": "synthetic-torch",
                "onnxruntime": "synthetic-ort",
                "numpy": "synthetic-numpy",
            },
        },
        "model": {
            "name": "mobilenet_v3_small",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "weights_sha256": "shared-weights-hash",
            "input_sha256": "shared-input-hash",
            "input_shape": [1, 3, 224, 224],
            "precision": "fp32",
            "model_seed": 2026,
            "input_seed": 2027,
        },
        "backend": backend_metadata,
        "threading": threading,
        "measurement_scope": {
            "name": "prepared_input_synchronized_call",
            "timed_operations": [
                "backend.run",
                "backend.synchronize",
            ],
            "request_concurrency": 1,
            "thread_policy": (
                "intra_op_swept_inter_op_runtime_default"
            ),
        },
        "correctness": {
            "before": {
                "allclose": True,
                "top1_match": True,
            },
            "after": {
                "allclose": True,
                "top1_match": True,
            },
        },
        "benchmark": {
            "warmup": 50,
            "iterations": 300,
        },
        # Repeated values are sufficient because these tests validate
        # result organization rather than percentile implementation.
        "samples_ns": [sample_ns] * 300,
        "statistics": {
            "count": 300,
            "mean_ms": mean_ms,
            "p50_ms": mean_ms,
            "p90_ms": mean_ms,
            "p99_ms": mean_ms,
            "min_ms": mean_ms,
            "max_ms": mean_ms,
            "std_ms": 0.0,
            "cv_percent": 0.0,
            "serial_service_rate_ips": 1000.0 / mean_ms,
        },
        "cpu_process_observation": {
            "scope": "complete_post_warmup_sampling_loop",
            "sampling_block_wall_time_ns": 1_000_000_000,
            "sampling_block_process_cpu_time_ns": (
                round_number * 1_000_000_000
            ),
            "average_process_cpu_core_equivalents": float(
                round_number
            ),
            "interpretation": (
                "aggregate_process_cpu_usage_"
                "not_active_thread_count"
            ),
        },
    }


def _complete_records() -> list[dict[str, Any]]:
    """Create the complete 3-backend, 5-setting, 3-round matrix."""

    return [
        _make_record(
            backend,
            intra_op_threads,
            round_number,
        )
        for backend in BACKENDS
        for intra_op_threads in INTRA_OP_VALUES
        for round_number in ROUNDS
    ]


def _set_field(
    record: dict[str, Any],
    path: str,
    value: Any,
) -> None:
    """Set one nested field in a synthetic result."""

    target = record
    keys = path.split(".")

    for key in keys[:-1]:
        target = target[key]

    target[keys[-1]] = value


def _find_group(
    summary: dict[str, Any],
    *,
    backend: str,
    intra_op_threads: int | None,
) -> dict[str, Any]:
    """Return one aggregate group from a validated summary."""

    matches = [
        group
        for group in summary["groups"]
        if group["backend"] == backend
        and group["requested_threads"]
        == intra_op_threads
    ]

    assert len(matches) == 1
    return matches[0]


def test_summarizer_validates_complete_matrix_and_aggregates_by_process():
    """Aggregate process summaries without pooling their raw samples."""

    summary = validate_and_summarize_thread_sweep(
        _complete_records()
    )

    assert summary["validation"]["record_count"] == 45
    assert summary["validation"]["group_count"] == 15
    assert summary["validation"]["status"] == "passed"

    group = _find_group(
        summary,
        backend="pytorch_cpu",
        intra_op_threads=2,
    )

    # Preserve the three independent process means as auditable evidence.
    run_means = [
        run["mean_ms"]
        for run in group["runs"]
    ]
    assert run_means == [10.0, 12.0, 14.0]

    # The representative result is the median of process-level means.
    assert group["representative_mean_ms"] == pytest.approx(
        12.0
    )

    # Relative range = (14 - 10) / mean(10, 12, 14) * 100.
    assert group[
        "cross_process_relative_range_percent"
    ] == pytest.approx(100.0 / 3.0)

    # Process CPU observations [1, 2, 3] also use their median.
    assert group[
        "representative_process_cpu_core_equivalents"
    ] == pytest.approx(2.0)


def test_summarizer_rejects_duplicate_backend_thread_round_case():
    """Reject two records claiming the same experiment coordinate."""

    records = _complete_records()
    records[-1] = copy.deepcopy(records[0])

    with pytest.raises(ValueError, match=r"Duplicate campaign case"):
        validate_and_summarize_thread_sweep(records)


def test_summarizer_rejects_missing_matrix_case():
    """Reject an incomplete campaign instead of silently summarizing it."""

    records = _complete_records()
    records.pop()

    with pytest.raises(
        ValueError,
        match=r"Expected exactly 45 records, received 44"
    ):
        validate_and_summarize_thread_sweep(records)


def test_summarizer_rejects_inconsistent_reproducibility_fingerprint():
    """Require every process to benchmark the same source, model, and input."""

    records = _complete_records()
    records[-1]["model"]["input_sha256"] = (
        "different-input-hash"
    )

    with pytest.raises(ValueError, match=r"fingerprint differs"):
        validate_and_summarize_thread_sweep(records)


def test_summarizer_rejects_runtime_thread_configuration_mismatch():
    """Reject a result whose runtime setting differs from its request."""

    records = _complete_records()

    target = next(
        record
        for record in records
        if record["backend"]["name"] == "ort_cpu"
        and record["threading"]["intra_op"][
            "requested_threads"
        ] == 2
    )

    # Keep the normalized value correct but corrupt native ORT metadata.
    target["backend"]["intra_op_num_threads"] = 4

    with pytest.raises(
        ValueError,
        match=r"native intra-op value does not match",
    ):
        validate_and_summarize_thread_sweep(records)


def test_summarizer_rejects_dirty_source_state():
    """Require formal runs to originate from a clean implementation commit."""

    records = _complete_records()
    records[0]["source"]["git_status"] = (
        " M src/inference_bench/benchmark.py"
    )

    with pytest.raises(ValueError, match=r"Git status is not clean"):
        validate_and_summarize_thread_sweep(records)


def test_summarizer_rejects_failed_correctness_gate():
    """Never aggregate latency from a numerically invalid run."""

    records = _complete_records()
    records[0]["correctness"]["after"]["allclose"] = False

    with pytest.raises(ValueError, match=r"Correctness allclose failed"):
        validate_and_summarize_thread_sweep(records)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", 2, "schema version"),
        ("experiment", "wrong_experiment", "experiment"),
        ("status", "failed", "status"),
        ("benchmark.warmup", 2, "benchmark configuration"),
        (
            "measurement_scope.thread_policy",
            "wrong_policy",
            "measurement scope",
        ),
        (
            "environment.thread_environment.OMP_NUM_THREADS",
            "4",
            "thread environment",
        ),
        (
            "environment.power_source_snapshot_after_run",
            "Now drawing from 'Battery Power'",
            "AC Power",
        ),
        (
            "cpu_process_observation."
            "average_process_cpu_core_equivalents",
            99.0,
            "CPU core equivalents",
        ),
        ("statistics.min_ms", 999.0, "Stored min_ms"),
        (
            "environment.hardware_model",
            "different-machine",
            "fingerprint differs",
        ),
    ],
)
def test_summarizer_rejects_invalid_formal_record(
    path,
    value,
    message,
):
    """Reject records that violate the formal experiment contract."""

    records = _complete_records()
    _set_field(records[0], path, value)

    with pytest.raises(ValueError, match=message):
        validate_and_summarize_thread_sweep(records)


def test_summarizer_keeps_mps_out_of_cpu_optimization_metrics():
    """Report MPS only as host-setting sensitivity evidence."""

    summary = validate_and_summarize_thread_sweep(
        _complete_records()
    )
    mps_groups = [
        group
        for group in summary["groups"]
        if group["backend"] == "pytorch_mps"
    ]

    assert all(
        "speedup_vs_explicit_1_thread_x" not in group
        for group in mps_groups
    )
    assert {
        item["backend"]
        for item in summary["default_vs_best_explicit"]
    } == {"pytorch_cpu", "ort_cpu"}
    assert (
        "observed_fastest_setting_label"
        in summary["mps_host_thread_sensitivity"]
    )
