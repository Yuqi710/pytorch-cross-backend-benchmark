"""Unit tests for Day 7 profiling-campaign aggregation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from inference_bench.profiling_campaign import (
    OPERATOR_CSV_FIELDS,
    profiling_operator_csv_rows,
    validate_and_summarize_profiling_campaign,
    validate_cross_campaign,
)


# These fixtures mirror the campaign dimensions and the runtime strings that
# ORT records for each public graph-optimization level.
GRAPH_LEVELS = ("disable", "basic", "extended", "all")
RUNTIME_LEVELS = {
    "disable": "GraphOptimizationLevel.ORT_DISABLE_ALL",
    "basic": "GraphOptimizationLevel.ORT_ENABLE_BASIC",
    "extended": "GraphOptimizationLevel.ORT_ENABLE_EXTENDED",
    "all": "GraphOptimizationLevel.ORT_ENABLE_ALL",
}


def _graph(path: str, sha256: str, node_count: int) -> dict[str, Any]:
    """Build a deterministic ONNX graph summary fixture."""

    return {
        "path": path,
        "sha256": sha256,
        "file_size_bytes": node_count * 100,
        "node_count": node_count,
        "initializer_count": 2,
        "opset_imports": {"ai.onnx": 18},
        "operator_counts": {"ai.onnx::Conv": node_count},
    }


def _base_record(
    backend: str,
    level: str | None,
    threads: int,
) -> dict[str, Any]:
    """Create one valid synthetic PyTorch or ORT profiling record."""

    run_id = f"{backend}-{level}-t{threads}"
    trace_sha256 = f"trace-{run_id}"

    # All records share fingerprint fields; backend and case-specific fields
    # are filled below to exercise the two profiler normalization paths.
    record = {
        "schema_version": 1,
        "experiment": "day7_cpu_operator_profile",
        "run_id": run_id,
        "timestamp_utc": "2026-10-27T00:00:00+00:00",
        "status": "passed",
        "profiling_scope": {
            "purpose": "operator_diagnostics_not_formal_latency",
            "summary_window": "active_inference_runs_only",
            "request_concurrency": 1,
            "input_policy": "same_prepared_synthetic_input_reused",
        },
        "profile_config": {
            "external_warmup_runs": 10,
            "active_runs": 10,
            "pytorch_schedule": None,
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        },
        "threading": {
            "intra_op": {
                "policy": "explicit",
                "requested_threads": threads,
                "runtime_reported_value": threads,
            },
            "inter_op": {
                "policy": "runtime_default",
                "requested_threads": None,
                "runtime_reported_value": 0,
            },
        },
        "graph_optimization_level": level,
        "environment": {
            "python": "3.11.14",
            "architecture": "arm64",
            "system": "Darwin",
            "macos_version": "26.0.1",
            "logical_cpu_count": 12,
            "cpu_model": "synthetic-cpu",
            "hardware_model": "synthetic-hardware",
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
                "torchvision": "synthetic-torchvision",
                "onnx": "synthetic-onnx",
                "onnxruntime": "synthetic-ort",
                "numpy": "synthetic-numpy",
            },
        },
        "source": {
            "git_commit": "implementation-commit",
            "git_status": "",
            "source_sha256": "profiling-source-hash",
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
            "original_onnx_graph": None,
        },
        "correctness": {
            "before": {"allclose": True, "top1_match": True},
            "after": {"allclose": True, "top1_match": True},
        },
        "trace": {
            "path": f"traces/{run_id}.json",
            "sha256": trace_sha256,
        },
        "optimized_graph": None,
    }

    if backend == "pytorch_cpu":
        # PyTorch contributes CPU-time, memory, and shape measurements.
        record["profile_config"].update(
            {
                "pytorch_schedule": {
                    "wait_steps": 1,
                    "warmup_steps": 2,
                    "active_steps": 10,
                    "repeat": 1,
                },
                "record_shapes": True,
                "profile_memory": True,
            }
        )
        record["threading"]["inter_op"][
            "runtime_reported_value"
        ] = 18
        record["backend"] = {
            "name": "pytorch_cpu",
            "runtime": "pytorch",
            "device": "cpu",
            "precision": "fp32",
            "torch_num_threads": threads,
            "torch_num_interop_threads": 18,
        }
        record["profile_summary"] = {
            "total_self_cpu_time_us": 100.0,
            "operator_group_count": 1,
            "operators": [
                {
                    "name": "aten::conv2d",
                    "call_count": 10,
                    "self_cpu_time_us": 100.0,
                    "total_cpu_time_us": 120.0,
                    "self_cpu_time_percent": 100.0,
                    "self_cpu_memory_bytes": 0,
                    "total_cpu_memory_bytes": 0,
                    "input_shapes": [],
                }
            ],
        }
    else:
        # ORT contributes node duration, provider, and graph metadata.
        original_graph = _graph("model.onnx", "onnx-hash", 10)
        record["model"]["original_onnx_graph"] = original_graph
        record["backend"] = {
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
            "profiling_enabled": True,
        }
        # The t1 session supplies one optimized graph for each ORT level.
        if threads == 1:
            record["optimized_graph"] = _graph(
                f"{level}.onnx",
                f"optimized-{level}",
                8,
            )
        record["profile_summary"] = {
            "trace_sha256": trace_sha256,
            "event_count": 20,
            "active_run_count": 10,
            "node_event_count": 10,
            "total_model_run_duration_us": 200,
            "total_node_duration_us": 180,
            "runtime_unattributed_duration_us": 20,
            "operators": [
                {
                    "operator": "Conv",
                    "provider": "CPUExecutionProvider",
                    "call_count": 10,
                    "total_duration_us": 180,
                    "max_memory_in_use_bytes": 100,
                    "max_memory_in_use_peak_bytes": 120,
                    "activation_bytes": 50,
                    "events_with_thread_scheduling_stats": 10,
                    "mean_duration_us": 18.0,
                    "node_duration_percent": 100.0,
                }
            ],
        }

    return record


def _complete_records() -> list[dict[str, Any]]:
    """Build the complete two-backend profiling campaign matrix."""

    return [
        _base_record("pytorch_cpu", None, threads)
        for threads in (1, 6)
    ] + [
        _base_record("ort_cpu", level, threads)
        for level in GRAPH_LEVELS
        for threads in (1, 6)
    ]


def test_profiling_campaign_validates_and_summarizes_ten_profiles():
    """Validate the full campaign and its operator and graph summaries."""

    operator_summary, graph_summary, fingerprint = (
        validate_and_summarize_profiling_campaign(
            _complete_records()
        )
    )

    assert operator_summary["validation"]["record_count"] == 10
    assert len(operator_summary["profiles"]) == 10
    assert len(operator_summary["operators"]) == 10
    assert graph_summary["validation"]["optimized_graph_count"] == 4
    assert fingerprint["git_commit"] == "implementation-commit"

    # Preserve the public column order when projecting operator rows to CSV.
    rows = profiling_operator_csv_rows(operator_summary)
    assert list(rows[0]) == list(OPERATOR_CSV_FIELDS)


def test_cross_campaign_validation_matches_shared_artifact_identity():
    """Match profiling evidence to the formal benchmark artifacts."""

    _, graph_summary, profile_fingerprint = (
        validate_and_summarize_profiling_campaign(
            _complete_records()
        )
    )
    # Copy the profiling campaign's shared identity fields to simulate the
    # fingerprint stored in a real graph-optimization summary.
    # Producer source hashes differ, so validation compares the shared
    # environment, workload identity, and original ONNX bytes.
    formal_fingerprint = {
        key: profile_fingerprint[key]
        for key in (
            "git_commit",
            "dependencies",
            "python",
            "architecture",
            "system",
            "macos_version",
            "logical_cpu_count",
            "cpu_model",
            "hardware_model",
            "memory_bytes",
            "thread_environment",
            "model_name",
            "model_weights",
            "weights_sha256",
            "input_sha256",
            "input_shape",
            "precision",
            "model_seed",
            "input_seed",
        )
    }
    formal_fingerprint["onnx_sha256"] = "onnx-hash"

    result = validate_cross_campaign(
        {"campaign": {"fingerprint": formal_fingerprint}},
        profile_fingerprint,
        graph_summary,
    )

    assert result == {
        "git_commit": "implementation-commit",
        "weights_sha256": "weights-hash",
        "input_sha256": "input-hash",
        "onnx_sha256": "onnx-hash",
        "status": "passed",
    }


def test_profiling_campaign_rejects_a_missing_case():
    with pytest.raises(ValueError, match="Expected exactly 10 profiling records"):
        # Take all but the final record, leaving nine of the ten configurations.
        validate_and_summarize_profiling_campaign(_complete_records()[:-1])


def test_profiling_campaign_rejects_a_duplicate_case():
    records = _complete_records()
    records[-1] = copy.deepcopy(records[0])

    with pytest.raises(ValueError, match="Duplicate profiling case"):
        validate_and_summarize_profiling_campaign(records)


def test_profiling_campaign_rejects_mixed_input_fingerprints():
    records = _complete_records()
    records[-1]["model"]["input_sha256"] = "different-input"

    with pytest.raises(ValueError, match="fingerprint differs"):
        validate_and_summarize_profiling_campaign(records)


def test_profiling_campaign_rejects_mixed_original_onnx_graphs():
    records = _complete_records()
    records[-1]["model"]["original_onnx_graph"]["sha256"] = "different-onnx"

    with pytest.raises(ValueError, match="Original ONNX graph differs"):
        validate_and_summarize_profiling_campaign(records)


def test_profiling_campaign_rejects_an_optimized_graph_from_t6():
    """Archive one optimized graph per level while profiling both thread counts.

    Both t1 and t6 apply the requested graph policy; only t1 serializes the graph.
    """
    records = _complete_records()
    records[-1]["optimized_graph"] = _graph("all-t6.onnx", "optimized-all", 8)

    with pytest.raises(ValueError, match="Only the ORT t1 profile"):
        validate_and_summarize_profiling_campaign(records)


def test_profiling_campaign_rejects_a_trace_hash_mismatch():
    """Check recorded trace identity without reopening the trace file."""

    records = _complete_records()
    records[-1]["profile_summary"]["trace_sha256"] = "different-trace"

    with pytest.raises(ValueError, match="active profile window"):
        validate_and_summarize_profiling_campaign(records)


def test_cross_campaign_validation_rejects_different_onnx_bytes():
    """Link formal latency and profiling evidence to the same ONNX export."""

    _, graph_summary, fingerprint = validate_and_summarize_profiling_campaign(
        _complete_records()
    )
    formal_fingerprint = {**fingerprint, "onnx_sha256": "different-onnx"}

    with pytest.raises(ValueError, match="different ONNX bytes"):
        validate_cross_campaign(
            {"campaign": {"fingerprint": formal_fingerprint}},
            fingerprint,
            graph_summary,
        )
