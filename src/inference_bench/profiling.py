"""Summarize PyTorch and ONNX Runtime CPU profiling artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of a file's exact serialized contents."""

    digest = hashlib.sha256()
    # Read in bounded blocks so that large models and traces do not need
    # to be loaded into memory as one byte string.
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_onnx_graph(
    model_path: str | Path,
) -> dict[str, object]:
    """Summarize the serialized structure of an ONNX graph.

    Operator counts include each node's domain so that standard ONNX
    operators remain distinguishable from runtime-specific operators
    introduced by ONNX Runtime graph optimization.
    """

    path = Path(model_path)
    model = onnx.load(str(path), load_external_data=True)

    # Standard ONNX nodes commonly serialize the default domain as an
    # empty string. Use "ai.onnx" as its readable summary label.
    operator_counts = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}"
        for node in model.graph.node
    )

    # Each imported operator domain has its own version. A model may,
    # for example, use standard ONNX opset 18 together with version 1
    # of the Microsoft runtime-specific operator domain.
    opset_imports = {
        opset.domain or "ai.onnx": int(opset.version)
        for opset in model.opset_import
    }

    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "file_size_bytes": path.stat().st_size,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "opset_imports": dict(sorted(opset_imports.items())),
        "operator_counts": dict(sorted(operator_counts.items())),
    }


def _domain_counts(
    operator_counts: dict[str, int],
) -> dict[str, int]:
    """Aggregate domain counts from domain-qualified operator names."""

    counts: Counter[str] = Counter()

    # Operator keys use ``domain::op_type``. Collapsing the operator
    # dimension exposes domain-level rewrites introduced by optimization.
    for operator, count in operator_counts.items():
        domain = operator.split("::", maxsplit=1)[0]
        counts[domain] += count

    # Stable ordering keeps serialized graph comparisons reproducible.
    return dict(sorted(counts.items()))


def _signed_changes(
    original: dict[str, int],
    optimized: dict[str, int],
) -> dict[str, int]:
    """Return nonzero optimized-minus-original count changes."""

    changes = {}

    # If an operator is missing from either graph, treat its count as zero
    # so newly added and removed operators are included in the comparison.
    for name in sorted(set(original) | set(optimized)):
        difference = optimized.get(name, 0) - original.get(name, 0)
        # Positive values are additions and negative values are removals.
        # Unchanged entries are omitted to keep the structural delta focused.
        if difference:
            changes[name] = difference

    return changes


def compare_onnx_graph_summaries(
    level: str,
    original: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    """Compare one optimized graph summary with the original graph."""

    original_operators = original["operator_counts"]
    optimized_operators = optimized["operator_counts"]

    # All absolute and relative changes use the exported original graph as
    # the baseline. These are structural diagnostics, not latency attribution.
    return {
        "graph_optimization_level": level,
        "graph": dict(optimized),
        "node_count_change": (
            optimized["node_count"] - original["node_count"]
        ),
        "node_count_change_percent": (
            optimized["node_count"] / original["node_count"] - 1
        ) * 100,
        "initializer_count_change": (
            optimized["initializer_count"]
            - original["initializer_count"]
        ),
        "file_size_change_bytes": (
            optimized["file_size_bytes"]
            - original["file_size_bytes"]
        ),
        "file_size_change_percent": (
            optimized["file_size_bytes"]
            / original["file_size_bytes"]
            - 1
        ) * 100,
        "operator_count_changes": _signed_changes(
            original_operators,
            optimized_operators,
        ),
        # Domain deltas complement operator deltas by revealing when ORT
        # introduces or removes runtime-specific operator domains.
        "domain_count_changes": _signed_changes(
            _domain_counts(original_operators),
            _domain_counts(optimized_operators),
        ),
    }


def summarize_torch_profile(
    profiler: Any,
) -> dict[str, object]:
    """Aggregate a completed PyTorch CPU profiler capture.

    Events are grouped by operator and input shape. Self CPU time is
    used for percentage calculations because total CPU time includes
    nested child operators and therefore cannot be summed without
    double-counting.
    """

    events = profiler.key_averages(
        group_by_input_shape=True
    )

    rows = []
    for event in events:
        rows.append(
            {
                "name": event.key,
                "call_count": int(event.count),
                "self_cpu_time_us": float(
                    event.self_cpu_time_total
                ),
                "total_cpu_time_us": float(
                    event.cpu_time_total
                ),
                "self_cpu_memory_bytes": int(
                    event.self_cpu_memory_usage
                ),
                "total_cpu_memory_bytes": int(
                    event.cpu_memory_usage
                ),
                "input_shapes": event.input_shapes,
            }
        )

    total_self_cpu_time_us = sum(
        max(row["self_cpu_time_us"], 0.0)
        for row in rows
    )

    for row in rows:
        row["self_cpu_time_percent"] = (
            row["self_cpu_time_us"]
            / total_self_cpu_time_us
            * 100
            if total_self_cpu_time_us
            else 0.0
        )

    # Place the largest exclusive CPU-time contributors first so that
    # the beginning of the serialized result is also the hotspot list.
    rows.sort(
        key=lambda row: row["self_cpu_time_us"],
        reverse=True,
    )

    return {
        "total_self_cpu_time_us": total_self_cpu_time_us,
        "operator_group_count": len(rows),
        "operators": rows,
    }


def _event_inside_run(
    event: dict[str, Any],
    run: dict[str, Any],
) -> bool:
    """Return whether one trace event is fully inside a model run."""

    event_start = int(event["ts"])
    event_end = event_start + int(event["dur"])
    run_start = int(run["ts"])
    run_end = run_start + int(run["dur"])

    return run_start <= event_start and event_end <= run_end


def _integer_argument(
    event: dict[str, Any],
    name: str,
) -> int:
    """Read an integer-valued field from an ORT event's arguments."""

    # A missing or empty field contributes zero to the aggregate.
    value = event.get("args", {}).get(name, 0)
    return int(value or 0)


def summarize_ort_profile(
    trace_path: str | Path,
    *,
    active_runs: int,
) -> dict[str, object]:
    """Summarize the active inference window of an ORT profile trace.

    The profiling script records correctness and warm-up calls before
    the active window, then ends profiling immediately after the active
    calls. The final ``active_runs`` model-run events therefore identify
    the inference calls included in this summary.

    Node events are grouped by operator type and execution provider.
    Their durations describe profiled wall-clock intervals rather than
    aggregate CPU time across worker threads.
    """

    path = Path(trace_path)
    events = json.loads(path.read_text(encoding="utf-8"))

    # Each Session/model_run event corresponds to one session.run()
    # call. Earlier runs may belong to correctness checking or warm-up.
    model_runs = [
        event
        for event in events
        if event.get("cat") == "Session"
        and event.get("name") == "model_run"
    ]

    # The caller ends profiling before the post-profile correctness
    # check, so the final runs are exactly the active inference window.
    selected_runs = model_runs[-active_runs:]

    if len(selected_runs) != active_runs:
        raise ValueError(
            f"Expected {active_runs} active model runs, "
            f"found {len(selected_runs)}"
        )

    # Retain only node executions fully contained in one selected
    # model-run interval. Initialization and warm-up nodes are excluded.
    node_events = [
        event
        for event in events
        if event.get("cat") == "Node"
        and any(
            _event_inside_run(event, run)
            for run in selected_runs
        )
    ]

    grouped: dict[tuple[str, str], dict[str, object]] = {}

    for event in node_events:
        arguments = event.get("args", {})
        key = (
            arguments.get("op_name", event.get("name", "unknown")),
            arguments.get("provider", "unknown"),
        )

        row = grouped.setdefault(
            key,
            {
                "operator": key[0],
                "provider": key[1],
                "call_count": 0,
                "total_duration_us": 0,
                "max_memory_in_use_bytes": 0,
                "max_memory_in_use_peak_bytes": 0,
                "activation_bytes": 0,
                "events_with_thread_scheduling_stats": 0,
            },
        )

        row["call_count"] += 1
        row["total_duration_us"] += int(event["dur"])

        # These fields describe runtime-wide memory state observed at
        # the event, rather than memory exclusively owned by the operator.
        row["max_memory_in_use_bytes"] = max(
            row["max_memory_in_use_bytes"],
            _integer_argument(event, "mem_bytes_in_use"),
        )
        row["max_memory_in_use_peak_bytes"] = max(
            row["max_memory_in_use_peak_bytes"],
            _integer_argument(event, "mem_in_use_peak"),
        )
        row["activation_bytes"] += _integer_argument(
            event,
            "activation_size",
        )
        if arguments.get("thread_scheduling_stats"):
            row["events_with_thread_scheduling_stats"] += 1

    total_model_run_duration_us = sum(
        int(run["dur"]) for run in selected_runs
    )
    total_node_duration_us = sum(
        int(event["dur"]) for event in node_events
    )

    rows = list(grouped.values())
    for row in rows:
        row["mean_duration_us"] = (
            row["total_duration_us"] / row["call_count"]
        )
        row["node_duration_percent"] = (
            row["total_duration_us"]
            / total_node_duration_us
            * 100
            if total_node_duration_us
            else 0.0
        )

    rows.sort(
        key=lambda row: row["total_duration_us"],
        reverse=True,
    )

    # The difference covers work visible in the enclosing model-run
    # interval but not attributed to Node events, such as executor and
    # runtime bookkeeping. It does not identify a specific root cause.
    runtime_unattributed_duration_us = max(
        total_model_run_duration_us - total_node_duration_us,
        0,
    )

    return {
        "trace_sha256": file_sha256(path),
        "event_count": len(events),
        "active_run_count": len(selected_runs),
        "node_event_count": len(node_events),
        "total_model_run_duration_us": (
            total_model_run_duration_us
        ),
        "total_node_duration_us": total_node_duration_us,
        "runtime_unattributed_duration_us": (
            runtime_unattributed_duration_us
        ),
        "operators": rows,
    }
