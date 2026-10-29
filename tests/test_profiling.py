"""Unit and integration tests for ONNX graph and CPU profiling summaries."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from inference_bench.ort_backend import create_cpu_session, run_ort_inference
from inference_bench.profiling import (
    compare_onnx_graph_summaries,
    summarize_onnx_graph,
    summarize_ort_profile,
    summarize_torch_profile,
)


def test_summarize_onnx_graph_counts_domains_and_fingerprints_bytes(tmp_path):
    """Verify graph structure, operator domains, and serialized file identity."""

    # Same-named operators in different domains must remain separate. The
    # synthetic custom-domain node is inspected without executing the model.
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "bias"], ["added"]),
            helper.make_node("Relu", ["added"], ["activated"]),
            helper.make_node("Relu", ["activated"], ["y"], domain="example"),
        ],
        "domain_counts",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[
            # One initializer tensor stores all three FP32 bias elements.
            helper.make_tensor("bias", TensorProto.FLOAT, [1, 3], [1, 2, -1])
        ],
    )
    model = helper.make_model(
        graph,
        # Each operator domain declares its own opset version.
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("example", 1)],
    )
    path = tmp_path / "domains.onnx"
    onnx.save(model, path)

    summary = summarize_onnx_graph(path)

    assert summary["path"] == path.as_posix()
    # Compute the expected digest directly from the saved file bytes.
    assert summary["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary["file_size_bytes"] == path.stat().st_size
    assert summary["node_count"] == 3
    assert summary["initializer_count"] == 1
    assert summary["opset_imports"] == {"ai.onnx": 18, "example": 1}
    assert summary["operator_counts"] == {
        "ai.onnx::Add": 1,
        "ai.onnx::Relu": 1,
        "example::Relu": 1,
    }


def test_compare_onnx_graph_summaries_reports_signed_operator_and_domain_changes():
    """Check signed structural changes against the original graph baseline."""

    # Handwritten summaries isolate comparison arithmetic from ORT graph rewrites.
    original = {
        "node_count": 4,
        "initializer_count": 3,
        "file_size_bytes": 1000,
        "operator_counts": {"ai.onnx::Conv": 2, "ai.onnx::Relu": 2},
    }
    optimized = {
        "node_count": 3,
        "initializer_count": 2,
        "file_size_bytes": 800,
        "operator_counts": {"ai.onnx::Conv": 2, "com.microsoft::FusedConv": 1},
    }

    comparison = compare_onnx_graph_summaries("extended", original, optimized)

    assert comparison["graph_optimization_level"] == "extended"
    assert comparison["graph"] == optimized
    assert comparison["node_count_change"] == -1
    assert comparison["node_count_change_percent"] == pytest.approx(-25)
    assert comparison["initializer_count_change"] == -1
    assert comparison["file_size_change_bytes"] == -200
    assert comparison["file_size_change_percent"] == pytest.approx(-20)
    # Omit unchanged Conv counts while retaining removed and added operators.
    assert comparison["operator_count_changes"] == {
        "ai.onnx::Relu": -2,
        "com.microsoft::FusedConv": 1,
    }
    assert comparison["domain_count_changes"] == {
        "ai.onnx": -2,
        "com.microsoft": 1,
    }


def test_summarize_torch_profile_preserves_time_memory_and_shapes():
    """Preserve PyTorch operator metrics and rank groups by self CPU time."""

    class FakeProfiler:
        """Supply deterministic operator groups through the profiler interface."""

        def key_averages(self, *, group_by_input_shape):
            """Return fixed aggregates for shape-aware summary tests."""

            assert group_by_input_shape is True
            # Times are already accumulated across each group's calls.
            # Negative memory deltas represent net release and must be preserved.
            return [
                SimpleNamespace(
                    key="aten::child", count=4,
                    self_cpu_time_total=25.0, cpu_time_total=25.0,
                    self_cpu_memory_usage=-32, cpu_memory_usage=-16,
                    input_shapes=[[2, 3]],
                ),
                SimpleNamespace(
                    key="aten::parent", count=2,
                    self_cpu_time_total=75.0, cpu_time_total=100.0,
                    self_cpu_memory_usage=64, cpu_memory_usage=96,
                    input_shapes=[[1, 3]],
                ),
            ]

    summary = summarize_torch_profile(FakeProfiler())

    # Exclusive times sum to 75 + 25 = 100 us. Adding inclusive times would
    # double-count the child contribution: 100 + 25 = 125 us.
    assert summary["total_self_cpu_time_us"] == 100.0
    assert summary["operator_group_count"] == 2
    # The larger self-time contributor comes first despite appearing second
    # in the profiler's input groups.
    assert summary["operators"] == [
        {
            "name": "aten::parent", "call_count": 2,
            "self_cpu_time_us": 75.0, "total_cpu_time_us": 100.0,
            "self_cpu_memory_bytes": 64, "total_cpu_memory_bytes": 96,
            "input_shapes": [[1, 3]], "self_cpu_time_percent": 75.0,
        },
        {
            "name": "aten::child", "call_count": 4,
            "self_cpu_time_us": 25.0, "total_cpu_time_us": 25.0,
            "self_cpu_memory_bytes": -32, "total_cpu_memory_bytes": -16,
            "input_shapes": [[2, 3]], "self_cpu_time_percent": 25.0,
        },
    ]


@pytest.fixture
def ort_trace_path(tmp_path):
    """Write a trace with warm-up, active, outside, and crossing events."""

    def node(ts, dur, operator, provider="CPUExecutionProvider", **args):
        """Build one synthetic node event with times in microseconds."""

        return {
            "cat": "Node", "name": f"{operator}_kernel_time",
            "ts": ts, "dur": dur,
            "args": {"op_name": operator, "provider": provider, **args},
        }

    # Model runs span [0, 1000], [2000, 2100], and [3000, 3150]. Selecting
    # the final two runs must exclude warm-up and nodes outside either window.
    trace = [
        {"cat": "Session", "name": "model_run", "ts": 0, "dur": 1000},
        node(100, 800, "Warmup"),
        node(1500, 30, "Outside"),
        {"cat": "Session", "name": "model_run", "ts": 2000, "dur": 100},
        node(2010, 40, "Conv", mem_bytes_in_use="100", mem_in_use_peak="150",
             activation_size="32"),
        node(2050, 30, "Conv", mem_bytes_in_use="50", mem_in_use_peak="200",
             activation_size="64", thread_scheduling_stats={"worker": 1}),
        # This event ends at 2105, beyond the second model run's end at 2100.
        node(2095, 10, "CrossBoundary"),
        {"cat": "Session", "name": "model_run", "ts": 3000, "dur": 150},
        node(3010, 50, "Conv", mem_bytes_in_use="120", mem_in_use_peak="180",
             activation_size="16", thread_scheduling_stats={"worker": 1}),
        # A synthetic second provider tests grouping without another runtime EP.
        node(3060, 20, "Conv", provider="OtherExecutionProvider",
             mem_bytes_in_use="40", mem_in_use_peak="60", activation_size="8"),
        node(3080, 10, "Relu"),
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def test_ort_summary_uses_last_active_runs(ort_trace_path):
    """Restrict the summary to the final requested model-run window."""

    summary = summarize_ort_profile(ort_trace_path, active_runs=1)

    # The final run lasts 150 us and contains nodes lasting 50 + 20 + 10 = 80 us,
    # leaving 70 us unattributed to node events.
    assert summary["active_run_count"] == 1
    assert summary["total_model_run_duration_us"] == 150
    assert summary["node_event_count"] == 3
    assert summary["total_node_duration_us"] == 80
    assert summary["runtime_unattributed_duration_us"] == 70
    assert {row["operator"] for row in summary["operators"]} == {"Conv", "Relu"}


def test_ort_summary_groups_by_operator_and_provider(ort_trace_path):
    """Aggregate active node events by both operator type and provider."""

    summary = summarize_ort_profile(ort_trace_path, active_runs=2)

    assert summary["trace_sha256"] == hashlib.sha256(
        ort_trace_path.read_bytes()
    ).hexdigest()
    # Raw event count includes three Session events and eight Node events.
    assert summary["event_count"] == 11
    assert summary["active_run_count"] == 2
    # Exclude Warmup, Outside, and CrossBoundary: eight nodes become five.
    assert summary["node_event_count"] == 5
    # Selected model runs total 100 + 150 = 250 us; selected nodes total 150 us.
    assert summary["total_model_run_duration_us"] == 250
    assert summary["total_node_duration_us"] == 150
    assert summary["runtime_unattributed_duration_us"] == 100
    # Each row represents one operator/provider group, sorted by total duration.
    rows = summary["operators"]
    assert [(row["operator"], row["provider"]) for row in rows] == [
        ("Conv", "CPUExecutionProvider"),
        ("Conv", "OtherExecutionProvider"),
        ("Relu", "CPUExecutionProvider"),
    ]
    # CPU Conv has three events: 40 + 30 + 50 = 120 us, a 40 us mean, and
    # 120 / 150 = 80% of node time. Runtime memory observations use maxima
    # (120, 200); activation sizes sum to 112. Two events carry scheduling stats.
    assert rows[0] == {
        "operator": "Conv", "provider": "CPUExecutionProvider",
        "call_count": 3, "total_duration_us": 120,
        "max_memory_in_use_bytes": 120, "max_memory_in_use_peak_bytes": 200,
        "activation_bytes": 112, "events_with_thread_scheduling_stats": 2,
        "mean_duration_us": 40.0, "node_duration_percent": 80.0,
    }
    assert rows[1]["total_duration_us"] == 20
    assert rows[1]["call_count"] == 1
    assert rows[1]["node_duration_percent"] == pytest.approx(100 / 7.5)
    # Missing optional memory observations retain the summary's zero default.
    assert rows[2]["max_memory_in_use_peak_bytes"] == 0


def test_ort_summary_rejects_an_incomplete_active_window(ort_trace_path):
    """Reject a trace with fewer model runs than the requested active window."""

    with pytest.raises(ValueError, match="Expected 4 active model runs"):
        summarize_ort_profile(ort_trace_path, active_runs=4)


def test_ort_summary_reads_a_real_profile_trace(tmp_path):
    """Parse a real CPU-only ORT trace from a small deterministic model."""

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["x", "bias"], ["added"]),
                helper.make_node("Relu", ["added"], ["y"]),
            ],
            "profile_model",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            initializer=[
                helper.make_tensor("bias", TensorProto.FLOAT, [1, 3], [1, -1, 0.5])
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    # Disable graph rewrites so each inference retains one Add and one Relu node.
    session = create_cpu_session(
        model_path,
        intra_op_num_threads=1,
        graph_optimization_level="disable",
        profile_file_prefix=tmp_path / "ort_profile",
    )
    x = torch.tensor([[-2.0, 1.0, 3.0]])
    # Record four ordinary calls; only the final two should enter the summary.
    for _ in range(4):
        output = run_ort_inference(session, x)
    # Close and flush profiling before reading the runtime-generated trace file.
    trace_path = Path(session.end_profiling())

    summary = summarize_ort_profile(trace_path, active_runs=2)

    # relu([-2, 1, 3] + [1, -1, 0.5]) = [0, 0, 3.5].
    torch.testing.assert_close(output, torch.tensor([[0.0, 0.0, 3.5]]))
    events = json.loads(trace_path.read_text(encoding="utf-8"))
    model_runs = [
        event for event in events
        if event.get("cat") == "Session" and event.get("name") == "model_run"
    ]
    assert len(model_runs) == 4
    assert summary["active_run_count"] == 2
    # Two selected calls execute two nodes each, producing four active events.
    assert summary["node_event_count"] == 4
    # Derive the expected duration from the two selected trace intervals.
    assert summary["total_model_run_duration_us"] == sum(
        run["dur"] for run in model_runs[-2:]
    )
    assert {row["operator"] for row in summary["operators"]} == {"Add", "Relu"}
    assert all(row["provider"] == "CPUExecutionProvider" for row in summary["operators"])
    assert all(row["call_count"] == 2 for row in summary["operators"])
