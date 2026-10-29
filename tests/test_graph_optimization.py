"""Unit tests for Day 7 graph-optimization validation and aggregation."""

import copy
from typing import Any

import pytest

from inference_bench.graph_optimization import (
    graph_optimization_csv_rows,
    validate_and_summarize_graph_optimization,
)


# Choose process means whose median differs clearly from their average. Vary
# thread scaling by graph level to detect comparisons of the wrong groups.
PROCESS_MEANS_MS = {
    ("disable", 1): (10.0, 12.0, 40.0),
    ("disable", 6): (4.0, 6.0, 20.0),
    ("basic", 1): (4.0, 6.0, 20.0),
    ("basic", 6): (2.0, 3.0, 10.0),
    ("extended", 1): (2.0, 3.0, 10.0),
    ("extended", 6): (1.0, 3.0, 8.0),
    ("all", 1): (1.0, 1.5, 5.0),
    ("all", 6): (2.0, 3.0, 9.0),
}
# Keep expected runtime strings independent of the production level mapping.
RUNTIME_LEVELS = {
    "disable": "GraphOptimizationLevel.ORT_DISABLE_ALL",
    "basic": "GraphOptimizationLevel.ORT_ENABLE_BASIC",
    "extended": "GraphOptimizationLevel.ORT_ENABLE_EXTENDED",
    "all": "GraphOptimizationLevel.ORT_ENABLE_ALL",
}


def _make_record(level, threads, round_number):
    """Create one valid synthetic process result."""

    mean_ms = PROCESS_MEANS_MS[(level, threads)][round_number - 1]
    # Process CPU observations also use a skewed series to verify their median.
    core_equivalents = (1.0, 2.0, 8.0)[round_number - 1]
    # Source, host, model, and input identity remain fixed across matrix cases.
    return {
        "schema_version": 1,
        "experiment": "day7_ort_graph_optimization",
        "run_id": f"{level}-t{threads}-r{round_number}",
        "timestamp_utc": "2026-10-28T00:00:00+00:00",
        "round": round_number,
        "status": "passed",
        "source": {
            "git_commit": "implementation-commit",
            "git_status": "",
            "source_sha256": "source-hash",
        },
        "environment": {
            "python": "3.11.16",
            "architecture": "arm64",
            "system": "Darwin",
            "macos_version": "26.5.1",
            "logical_cpu_count": 18,
            "cpu_model": "synthetic-cpu",
            "hardware_model": "synthetic-hardware",
            "memory_bytes": "51539607552",
            "power_source_snapshot_after_run": "Now drawing from 'AC Power'",
            "thread_environment": {
                "OMP_NUM_THREADS": None,
                "MKL_NUM_THREADS": None,
                "OPENBLAS_NUM_THREADS": None,
                "VECLIB_MAXIMUM_THREADS": None,
                "NUMEXPR_NUM_THREADS": None,
            },
            "versions": {
                "torch": "synthetic-torch",
                "torchvision": "synthetic-torchvision",
                "onnx": "synthetic-onnx",
                "onnxruntime": "synthetic-ort",
                "numpy": "synthetic-numpy",
            },
        },
        "model": {
            "name": "mobilenet_v3_small",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "weights_sha256": "weights-hash",
            "input_sha256": "input-hash",
            "input_shape": [1, 3, 224, 224],
            "precision": "fp32",
            "model_seed": 2026,
            "input_seed": 2027,
            "onnx": {
                "path": "model.onnx",
                "opset_version": 18,
                "sha256": "onnx-hash",
                "file_size_bytes": 1000,
            },
        },
        "measurement_scope": {
            "name": "prepared_input_synchronized_call",
            "request_concurrency": 1,
            "thread_policy": "explicit_intra_op_inter_op_runtime_default",
            "graph_optimization_policy": "one_level_per_independent_process",
        },
        "threading": {
            "experiment_variable": "intra_op_threads",
            "intra_op": {
                "policy": "explicit",
                "scope": "onnxruntime_cpu_operator_intra_op",
                "requested_threads": threads,
                "runtime_reported_value": threads,
            },
            "inter_op": {
                "policy": "runtime_default",
                "requested_threads": None,
                "runtime_reported_value": 0,
            },
        },
        "graph_optimization": {
            "requested_level": level,
            "runtime_level_name": level,
            "runtime_level": RUNTIME_LEVELS[level],
            "execution_mode": "ExecutionMode.ORT_SEQUENTIAL",
        },
        "backend": {
            "name": "ort_cpu",
            "runtime": "onnxruntime",
            "device": "cpu",
            "precision": "fp32",
            "providers": ["CPUExecutionProvider"],
            "graph_optimization_level_name": level,
            "graph_optimization_level": RUNTIME_LEVELS[level],
            "execution_mode": "ExecutionMode.ORT_SEQUENTIAL",
            "intra_op_num_threads": threads,
            "inter_op_num_threads": 0,
        },
        "diagnostic_features": {
            "profiling_enabled": False,
            "optimized_model_serialization_enabled": False,
        },
        "correctness": {
            "before": {"allclose": True, "top1_match": True},
            "after": {"allclose": True, "top1_match": True},
        },
        "benchmark": {"warmup": 50, "iterations": 300},
        # Repeated values give independently calculable process statistics;
        # these fixtures exercise campaign aggregation rather than percentiles.
        "samples_ns": [round(mean_ms * 1_000_000)] * 300,
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
            # CPU time divided by wall time represents average core usage,
            # not the number of active worker threads.
            "sampling_block_wall_time_ns": 1_000_000_000,
            "sampling_block_process_cpu_time_ns": int(core_equivalents * 1_000_000_000),
            "average_process_cpu_core_equivalents": core_equivalents,
        },
    }


@pytest.fixture
def records():
    """Create a fresh four-level, two-thread, three-round campaign matrix."""

    return [
        _make_record(level, threads, round_number)
        for level, threads in PROCESS_MEANS_MS
        for round_number in (1, 2, 3)
    ]


def _set_field(record: dict[str, Any], path: str, value: Any):
    """Set one nested field in a synthetic process result."""

    keys = path.split(".")
    for key in keys[:-1]:
        record = record[key]
    record[keys[-1]] = value


def test_graph_summary_preserves_processes_and_uses_median_metrics(records):
    """Aggregate process metrics without pooling or modifying raw evidence."""

    original_records = copy.deepcopy(records)

    summary = validate_and_summarize_graph_optimization(list(reversed(records)))

    assert records == original_records
    assert summary["validation"]["record_count"] == 24
    assert summary["validation"]["group_count"] == 8
    assert all(group["process_count"] == 3 for group in summary["groups"])
    # Groups follow the fixed graph-level and thread-count order, so disable/t1
    # remains first even when the input records are reversed.
    group = summary["groups"][0]
    assert (group["graph_optimization_level"], group["intra_op_threads"]) == ("disable", 1)
    assert [run["round"] for run in group["runs"]] == [1, 2, 3]
    assert group["process_mean_ms"] == [10.0, 12.0, 40.0]
    # The representative mean is median(10, 12, 40) = 12 ms;
    # the mean of the pooled samples would be approximately 20.67 ms.
    assert group["representative_mean_ms"] == pytest.approx(12.0)
    assert group["min_process_mean_ms"] == 10.0
    assert group["max_process_mean_ms"] == 40.0
    # Relative range = (40 - 10) / median(10, 12, 40) * 100 = 250%.
    assert group["cross_process_range_percent"] == pytest.approx(250.0)
    # Upper-tail summaries use process-level percentiles rather than the
    # percentile of all 900 samples pooled across processes.
    assert group["representative_p50_ms"] == pytest.approx(12.0)
    assert group["representative_p90_ms"] == pytest.approx(12.0)
    assert group["representative_p99_ms"] == pytest.approx(12.0)
    assert group["representative_core_equivalents"] == pytest.approx(2.0)


def test_graph_summary_compares_graph_levels_at_equal_threads(records):
    """Compare each graph policy with disabled optimization at equal threads."""

    summary = validate_and_summarize_graph_optimization(records)
    # Expected speedups and latency changes are keyed by both matrix variables
    # so a t1 baseline cannot silently substitute for the corresponding t6 case.
    expected = {
        ("disable", 1): (1.0, 0.0),
        ("disable", 6): (1.0, 0.0),
        ("basic", 1): (2.0, -50.0),
        ("basic", 6): (2.0, -50.0),
        ("extended", 1): (4.0, -75.0),
        ("extended", 6): (2.0, -50.0),
        ("all", 1): (8.0, -87.5),
        ("all", 6): (2.0, -50.0),
    }

    for group in summary["groups"]:
        speedup, latency_change = expected[
            (group["graph_optimization_level"], group["intra_op_threads"])
        ]
        assert group["speedup_vs_disable"] == pytest.approx(speedup)
        assert group["latency_change_vs_disable_percent"] == pytest.approx(latency_change)


def test_graph_summary_compares_threads_within_each_graph_level(records):
    """Measure t1-to-t6 scaling while keeping graph policy fixed."""

    summary = validate_and_summarize_graph_optimization(records)
    # Include a speedup, unchanged latency, and a slowdown. More threads do not
    # imply faster inference for every graph configuration.
    expected = {
        "disable": (12.0, 6.0, 0.5, 2.0),
        "basic": (6.0, 3.0, 0.5, 2.0),
        "extended": (3.0, 3.0, 1.0, 1.0),
        "all": (1.5, 3.0, 2.0, 0.5),
    }

    assert len(summary["thread_scaling_t1_to_t6"]) == 4
    for comparison in summary["thread_scaling_t1_to_t6"]:
        t1_mean, t6_mean, latency_ratio, speedup = expected[
            comparison["graph_optimization_level"]
        ]
        assert comparison["t1_representative_mean_ms"] == pytest.approx(t1_mean)
        assert comparison["t6_representative_mean_ms"] == pytest.approx(t6_mean)
        assert comparison["latency_ratio"] == pytest.approx(latency_ratio)
        assert comparison["speedup"] == pytest.approx(speedup)


def test_graph_csv_retains_round_means_and_scaling(records):
    """Preserve process-level evidence and derived comparisons in CSV rows."""

    rows = graph_optimization_csv_rows(
        validate_and_summarize_graph_optimization(records)
    )

    # CSV keeps the three process means alongside the representative metric
    # and the graph-level thread comparison for each configuration.
    assert len(rows) == 8
    assert [rows[0][f"round_{number}_mean_ms"] for number in (1, 2, 3)] == [10.0, 12.0, 40.0]
    assert rows[0]["representative_mean_ms"] == pytest.approx(12.0)
    assert rows[0]["t6_to_t1_latency_ratio"] == pytest.approx(0.5)
    assert rows[0]["t1_to_t6_speedup"] == pytest.approx(2.0)


def test_graph_summary_rejects_a_missing_case(records):
    """Reject an incomplete campaign instead of reporting a partial matrix."""

    with pytest.raises(ValueError, match="Expected exactly 24 records"):
        validate_and_summarize_graph_optimization(records[:-1])


def test_graph_summary_rejects_a_duplicate_case(records):
    """Reject two records claiming the same graph-level/thread/round case."""

    # Preserve the record count so rejection depends on case uniqueness.
    records[-1] = copy.deepcopy(records[0])

    with pytest.raises(ValueError, match="Duplicate campaign case"):
        validate_and_summarize_graph_optimization(records)


@pytest.mark.parametrize(
    "path",
    [
        "source.git_commit",
        "source.source_sha256",
        "model.weights_sha256",
        "model.input_sha256",
        "model.onnx.sha256",
        "environment.hardware_model",
    ],
)
def test_graph_summary_rejects_mixed_fingerprints(records, path):
    """Require a shared source, environment, model, input, and ONNX identity."""

    _set_field(records[-1], path, "different-value")

    with pytest.raises(ValueError, match="fingerprint differs"):
        validate_and_summarize_graph_optimization(records)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("status", "failed", "status"),
        ("source.git_status", " M benchmark.py", "Git status"),
        ("backend.providers", ["CoreMLExecutionProvider"], "CPUExecutionProvider"),
        ("backend.execution_mode", "ExecutionMode.ORT_PARALLEL", "sequential"),
        ("backend.intra_op_num_threads", 6, "intra-op metadata"),
        ("backend.graph_optimization_level", RUNTIME_LEVELS["all"], "runtime graph level"),
        ("diagnostic_features.profiling_enabled", True, "diagnostic features"),
        ("diagnostic_features.optimized_model_serialization_enabled", True, "diagnostic features"),
        ("correctness.before.allclose", False, "Correctness allclose"),
        ("correctness.after.top1_match", False, "Correctness Top-1"),
        ("statistics.mean_ms", 999.0, "Stored mean_ms"),
    ],
)
def test_graph_summary_rejects_invalid_formal_records(records, path, value, message):
    """Reject records that violate the formal experiment contract."""

    # Corrupt one field at a time while retaining a complete campaign matrix.
    _set_field(records[0], path, value)

    with pytest.raises(ValueError, match=message):
        validate_and_summarize_graph_optimization(records)


@pytest.mark.parametrize("samples", [[1_000_000] * 299, [0] * 300, [1.5] * 300])
def test_graph_summary_rejects_invalid_raw_samples(records, samples):
    """Require 300 positive integer nanosecond observations per process."""

    records[0]["samples_ns"] = samples

    with pytest.raises(ValueError, match="300 positive integer samples"):
        validate_and_summarize_graph_optimization(records)
