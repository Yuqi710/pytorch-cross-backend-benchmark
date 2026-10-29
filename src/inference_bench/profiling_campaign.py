"""Validate and summarize the Day 7 CPU profiling campaign."""

from __future__ import annotations

from typing import Any, Mapping

from inference_bench.ort_backend import GRAPH_OPTIMIZATION_LEVELS
from inference_bench.profiling import compare_onnx_graph_summaries
from inference_bench.result_validation import (
    field,
    finite_number,
    required_string,
    validate_ac_power,
    validate_correctness,
    validate_thread_environment,
)


GRAPH_LEVELS = tuple(GRAPH_OPTIMIZATION_LEVELS)

# The campaign contains two PyTorch profiles and the Cartesian product of
# four ORT optimization levels with two explicit intra-op thread counts.
PROFILE_CASES = (
    ("pytorch_cpu", None, 1),
    ("pytorch_cpu", None, 6),
    *tuple(
        ("ort_cpu", level, threads)
        for level in GRAPH_LEVELS
        for threads in (1, 6)
    ),
)

# One sparse schema represents both profilers. Backend-specific columns remain
# empty when the other profiler has no equivalent measurement.
OPERATOR_CSV_FIELDS = (
    "backend",
    "graph_optimization_level",
    "intra_op_threads",
    "operator",
    "provider",
    "call_count",
    "self_cpu_time_us",
    "total_cpu_time_us",
    "self_cpu_time_percent",
    "total_duration_us",
    "node_duration_percent",
    "self_cpu_memory_bytes",
    "total_cpu_memory_bytes",
    "max_memory_in_use_peak_bytes",
)


def validate_profile_environment(
    record: Mapping[str, Any],
) -> None:
    """Validate source cleanliness, AC power, and thread environment."""

    # A clean source tree and controlled runtime environment bind each trace
    # to the implementation and thread policy recorded by the campaign.
    git_status = field(record, "source.git_status")
    if not isinstance(git_status, str) or git_status.strip():
        raise ValueError("Profiling record has a dirty Git status")

    validate_thread_environment(record)
    validate_ac_power(record)


def validate_graph_summary(
    graph: Any,
    path: str,
) -> dict[str, Any]:
    """Validate one recorded ONNX graph summary."""

    # Validate the graph summary stored in the profiling result JSON
    # without reopening the archived ONNX model file.
    if not isinstance(graph, Mapping):
        raise ValueError(f"{path} must be an object")

    required_string(graph.get("path"), f"{path}.path")
    required_string(graph.get("sha256"), f"{path}.sha256")

    for name in (
        "file_size_bytes",
        "node_count",
        "initializer_count",
    ):
        value = graph.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"{path}.{name} must be a nonnegative integer"
            )

    if not isinstance(graph.get("opset_imports"), Mapping):
        raise ValueError(f"{path}.opset_imports must be an object")
    if not isinstance(graph.get("operator_counts"), Mapping):
        raise ValueError(f"{path}.operator_counts must be an object")

    for operator, count in graph["operator_counts"].items():
        if (
            not isinstance(operator, str)
            or type(count) is not int
            or count < 0
        ):
            raise ValueError(
                f"{path}.operator_counts contains an invalid value"
            )

    return dict(graph)


def profile_fingerprint(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract fields shared by all ten profiling processes."""

    # Backend, graph level, and thread count are campaign variables. The
    # fingerprint keeps only the source, environment, workload, and capture
    # settings that must remain invariant across independent processes.
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
        "hardware_model": field(
            record,
            "environment.hardware_model",
        ),
        "memory_bytes": field(record, "environment.memory_bytes"),
        "thread_environment": field(
            record,
            "environment.thread_environment",
        ),
        "profiling_scope": field(record, "profiling_scope"),
        "model_name": field(record, "model.name"),
        "model_weights": field(record, "model.weights"),
        "weights_sha256": field(
            record,
            "model.weights_sha256",
        ),
        "input_sha256": field(record, "model.input_sha256"),
        "input_shape": field(record, "model.input_shape"),
        "precision": field(record, "model.precision"),
        "model_seed": field(record, "model.model_seed"),
        "input_seed": field(record, "model.input_seed"),
        "external_warmup_runs": field(
            record,
            "profile_config.external_warmup_runs",
        ),
        "active_runs": field(
            record,
            "profile_config.active_runs",
        ),
    }


def common_operator_row(
    backend: str,
    level: str | None,
    threads: int,
    operator: str,
    provider: str | None,
    call_count: int,
) -> dict[str, Any]:
    """Create one unified operator-summary row."""

    # Start with the complete union of PyTorch and ORT metrics so unavailable
    # measurements have an explicit null value in JSON and a blank CSV cell.
    return {
        "backend": backend,
        "graph_optimization_level": level,
        "intra_op_threads": threads,
        "operator": operator,
        "provider": provider,
        "call_count": call_count,
        "self_cpu_time_us": None,
        "total_cpu_time_us": None,
        "self_cpu_time_percent": None,
        "total_duration_us": None,
        "node_duration_percent": None,
        "self_cpu_memory_bytes": None,
        "total_cpu_memory_bytes": None,
        "max_memory_in_use_bytes": None,
        "max_memory_in_use_peak_bytes": None,
        "activation_bytes": None,
        "events_with_thread_scheduling_stats": None,
        "mean_duration_us": None,
        "input_shapes": None,
    }


def validate_profile_record(
    record: Mapping[str, Any],
) -> tuple[
    tuple[str, str | None, int],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Validate and normalize one PyTorch or ORT profile record."""

    # Validate run identity, trace identity, and campaign-wide controls before
    # interpreting fields whose meaning depends on the profiler backend.
    if field(record, "schema_version") != 1:
        raise ValueError("Profiling schema version must be 1")
    if field(record, "experiment") != "day7_cpu_operator_profile":
        raise ValueError("Invalid Day 7 profiling experiment name")
    if field(record, "status") != "passed":
        raise ValueError("Profiling status must be passed")

    run_id = required_string(field(record, "run_id"), "run_id")
    timestamp = required_string(
        field(record, "timestamp_utc"),
        "timestamp_utc",
    )
    required_string(
        field(record, "source.git_commit"),
        "source.git_commit",
    )
    required_string(
        field(record, "source.source_sha256"),
        "source.source_sha256",
    )
    required_string(
        field(record, "trace.path"),
        "trace.path",
    )
    trace_sha256 = required_string(
        field(record, "trace.sha256"),
        "trace.sha256",
    )

    validate_correctness(record)
    validate_profile_environment(record)

    if (
        field(record, "profile_config.external_warmup_runs") != 10
        or field(record, "profile_config.active_runs") != 10
        or field(record, "profile_config.with_stack") is not False
    ):
        raise ValueError("Profiling configuration must use 10/10 runs")

    backend = field(record, "backend.name")
    level = field(record, "graph_optimization_level")
    threads = field(
        record,
        "threading.intra_op.requested_threads",
    )

    if backend not in ("pytorch_cpu", "ort_cpu"):
        raise ValueError(f"Invalid profiling backend: {backend}")
    if type(threads) is not int or threads not in (1, 6):
        raise ValueError(f"Invalid profiling thread count: {threads}")

    if (
        field(record, "threading.intra_op.policy") != "explicit"
        or field(
            record,
            "threading.intra_op.runtime_reported_value",
        )
        != threads
    ):
        raise ValueError("Profiling intra-op metadata is inconsistent")

    if (
        field(record, "threading.inter_op.policy") != "runtime_default"
        or field(
            record,
            "threading.inter_op.requested_threads",
        )
        is not None
    ):
        raise ValueError(
            "Profiling inter-op policy must remain runtime default"
        )

    profile_summary = field(record, "profile_summary")
    if not isinstance(profile_summary, Mapping):
        raise ValueError("profile_summary must be an object")

    operators = profile_summary.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("profile_summary.operators must be nonempty")

    original_graph: dict[str, Any] | None = None
    optimized_graph: dict[str, Any] | None = None
    normalized_operators: list[dict[str, Any]] = []

    if backend == "pytorch_cpu":
        # PyTorch reports exclusive and inclusive CPU time plus allocator
        # deltas for operator groups captured by its scheduled active window.
        if level is not None:
            raise ValueError(
                "PyTorch profile must not report an ORT graph level"
            )
        if field(record, "backend.runtime") != "pytorch":
            raise ValueError("Invalid PyTorch runtime metadata")
        if (
            field(record, "backend.device") != "cpu"
            or field(record, "backend.precision") != "fp32"
        ):
            raise ValueError("PyTorch profile must use CPU FP32")

        if (
            field(record, "backend.torch_num_threads")
            != threads
            or field(
                record,
                "threading.inter_op.runtime_reported_value",
            )
            != field(
                record,
                "backend.torch_num_interop_threads",
            )
        ):
            raise ValueError("PyTorch runtime thread metadata differs")

        if (
            field(record, "profile_config.record_shapes") is not True
            or field(record, "profile_config.profile_memory") is not True
            or field(record, "profile_config.pytorch_schedule")
            != {
                "wait_steps": 1,
                "warmup_steps": 2,
                "active_steps": 10,
                "repeat": 1,
            }
        ):
            raise ValueError("Invalid PyTorch profiler configuration")

        if field(record, "model.original_onnx_graph") is not None:
            raise ValueError(
                "PyTorch profile must not contain an ONNX graph"
            )
        if field(record, "optimized_graph") is not None:
            raise ValueError(
                "PyTorch profile must not contain an optimized graph"
            )

        if (
            profile_summary.get("operator_group_count")
            != len(operators)
        ):
            raise ValueError(
                "PyTorch operator_group_count is inconsistent"
            )

        finite_number(
            profile_summary.get("total_self_cpu_time_us"),
            "profile_summary.total_self_cpu_time_us",
        )

        for operator in operators:
            if not isinstance(operator, Mapping):
                raise ValueError(
                    "PyTorch operator row must be an object"
                )

            name = required_string(
                operator.get("name"),
                "profile_summary.operators.name",
            )
            call_count = operator.get("call_count")
            if type(call_count) is not int or call_count < 1:
                raise ValueError(
                    "PyTorch operator call_count must be positive"
                )

            row = common_operator_row(
                backend,
                None,
                threads,
                name,
                None,
                call_count,
            )
            row.update(
                {
                    "self_cpu_time_us": finite_number(
                        operator.get("self_cpu_time_us"),
                        "operator.self_cpu_time_us",
                    ),
                    "total_cpu_time_us": finite_number(
                        operator.get("total_cpu_time_us"),
                        "operator.total_cpu_time_us",
                    ),
                    "self_cpu_time_percent": finite_number(
                        operator.get("self_cpu_time_percent"),
                        "operator.self_cpu_time_percent",
                    ),
                    "self_cpu_memory_bytes": operator.get(
                        "self_cpu_memory_bytes"
                    ),
                    "total_cpu_memory_bytes": operator.get(
                        "total_cpu_memory_bytes"
                    ),
                    "input_shapes": operator.get("input_shapes"),
                }
            )
            normalized_operators.append(row)

    else:
        # ORT reports node-event duration and execution-provider attribution;
        # these values are not interchangeable with PyTorch CPU-time metrics.
        if level not in GRAPH_LEVELS:
            raise ValueError(f"Invalid ORT graph level: {level}")
        if field(record, "backend.runtime") != "onnxruntime":
            raise ValueError("Invalid ORT runtime metadata")
        if (
            field(record, "backend.device") != "cpu"
            or field(record, "backend.precision") != "fp32"
            or field(record, "backend.providers") != ["CPUExecutionProvider"]
        ):
            raise ValueError("ORT profile must use CPUExecutionProvider")

        if (
            field(record, "backend.graph_optimization_level_name")
            != level
            or field(record, "backend.graph_optimization_level")
            != str(GRAPH_OPTIMIZATION_LEVELS[level])
            or field(record, "backend.execution_mode")
            != "ExecutionMode.ORT_SEQUENTIAL"
            or field(record, "backend.intra_op_num_threads") != threads
            or field(record, "backend.inter_op_num_threads") != 0
            or field(record, "backend.profiling_enabled") is not True
        ):
            raise ValueError("ORT profiling metadata is inconsistent")

        if (
            field(record, "threading.inter_op.runtime_reported_value")
            != 0
        ):
            raise ValueError("ORT inter-op runtime value must be zero")

        if (
            field(record, "profile_config.record_shapes") is not False
            or field(record, "profile_config.profile_memory") is not False
            or field(record, "profile_config.pytorch_schedule") is not None
        ):
            raise ValueError("Invalid ORT profiling configuration")

        original_graph = validate_graph_summary(
            field(record, "model.original_onnx_graph"),
            "model.original_onnx_graph",
        )

        recorded_optimized_graph = field(
            record,
            "optimized_graph",
        )
        # One t1 session per level serializes the optimized graph. The t6
        # session contributes trace data without a second optimized artifact.
        if threads == 1:
            optimized_graph = validate_graph_summary(
                recorded_optimized_graph,
                "optimized_graph",
            )
        elif recorded_optimized_graph is not None:
            raise ValueError(
                "Only the ORT t1 profile may serialize an optimized graph"
            )

        if (
            profile_summary.get("trace_sha256") != trace_sha256
            or profile_summary.get("active_run_count") != 10
        ):
            raise ValueError("ORT active profile window is inconsistent")

        for name in (
            "event_count",
            "node_event_count",
            "total_model_run_duration_us",
            "total_node_duration_us",
            "runtime_unattributed_duration_us",
        ):
            value = profile_summary.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"profile_summary.{name} must be nonnegative"
                )

        for operator in operators:
            if not isinstance(operator, Mapping):
                raise ValueError("ORT operator row must be an object")

            name = required_string(
                operator.get("operator"),
                "profile_summary.operators.operator",
            )
            provider = required_string(
                operator.get("provider"),
                "profile_summary.operators.provider",
            )
            call_count = operator.get("call_count")
            if type(call_count) is not int or call_count < 1:
                raise ValueError(
                    "ORT operator call_count must be positive"
                )

            row = common_operator_row(
                backend,
                level,
                threads,
                name,
                provider,
                call_count,
            )
            row.update(
                {
                    "total_duration_us": operator.get(
                        "total_duration_us"
                    ),
                    "node_duration_percent": finite_number(
                        operator.get("node_duration_percent"),
                        "operator.node_duration_percent",
                    ),
                    "max_memory_in_use_bytes": operator.get(
                        "max_memory_in_use_bytes"
                    ),
                    "max_memory_in_use_peak_bytes": operator.get(
                        "max_memory_in_use_peak_bytes"
                    ),
                    "activation_bytes": operator.get(
                        "activation_bytes"
                    ),
                    "events_with_thread_scheduling_stats": (
                        operator.get(
                            "events_with_thread_scheduling_stats"
                        )
                    ),
                    "mean_duration_us": finite_number(
                        operator.get("mean_duration_us"),
                        "operator.mean_duration_us",
                    ),
                }
            )
            normalized_operators.append(row)

    # Keep per-profile totals separate from normalized operator rows so the
    # same operator payload is not duplicated in the campaign summary.
    profile_totals = {
        key: value
        for key, value in profile_summary.items()
        if key != "operators"
    }

    profile_entry = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "backend": backend,
        "graph_optimization_level": level,
        "intra_op_threads": threads,
        "active_runs": 10,
        "trace": dict(field(record, "trace")),
        "optimized_graph_sha256": (
            optimized_graph["sha256"]
            if optimized_graph is not None
            else None
        ),
        "profile_totals": profile_totals,
    }

    return (
        (backend, level, threads),
        profile_entry,
        normalized_operators,
        original_graph,
        optimized_graph,
    )


def validate_and_summarize_profiling_campaign(
    records: list[dict[str, object]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate ten profiles and build operator and graph summaries."""

    # Reject an incomplete campaign before selecting a reference fingerprint
    # or indexing records by their experimental configuration.
    if len(records) != len(PROFILE_CASES):
        raise ValueError(
            f"Expected exactly {len(PROFILE_CASES)} profiling records, "
            f"received {len(records)}"
        )

    reference_fingerprint = profile_fingerprint(records[0])
    indexed: dict[
        tuple[str, str | None, int],
        tuple[
            dict[str, Any],
            list[dict[str, Any]],
            dict[str, Any] | None,
            dict[str, Any] | None,
        ],
    ] = {}
    run_ids: set[str] = set()

    for record in records:
        # Every process must describe the same source, host, workload, and
        # profiling window; only the declared campaign variables may differ.
        if profile_fingerprint(record) != reference_fingerprint:
            raise ValueError(
                "Profiling source, environment, model, or input "
                "fingerprint differs"
            )

        (
            case,
            profile_entry,
            operator_rows,
            original_graph,
            optimized_graph,
        ) = validate_profile_record(record)

        if case in indexed:
            raise ValueError(f"Duplicate profiling case: {case}")
        if profile_entry["run_id"] in run_ids:
            raise ValueError(
                f"Duplicate profiling run_id: {profile_entry['run_id']}"
            )

        indexed[case] = (
            profile_entry,
            operator_rows,
            original_graph,
            optimized_graph,
        )
        run_ids.add(profile_entry["run_id"])

    if set(indexed) != set(PROFILE_CASES):
        missing = set(PROFILE_CASES) - set(indexed)
        unexpected = set(indexed) - set(PROFILE_CASES)
        raise ValueError(
            f"Incomplete profiling campaign: missing={missing}, "
            f"unexpected={unexpected}"
        )

    profiles = []
    operators = []
    original_graph: dict[str, Any] | None = None
    optimized_graphs: dict[str, dict[str, Any]] = {}

    # Canonical case order makes JSON and CSV output stable regardless of raw
    # input filename order.
    for case in PROFILE_CASES:
        (
            profile_entry,
            operator_rows,
            case_original_graph,
            case_optimized_graph,
        ) = indexed[case]

        profiles.append(profile_entry)
        operators.extend(operator_rows)

        if case_original_graph is not None:
            # All eight ORT sessions must reference the same original export.
            if original_graph is None:
                original_graph = case_original_graph
            elif case_original_graph != original_graph:
                raise ValueError(
                    "Original ONNX graph differs across ORT profiles"
                )

        backend, level, threads = case
        if (
            backend == "ort_cpu"
            and threads == 1
            and case_optimized_graph is not None
        ):
            # The four representative t1 sessions supply one optimized graph
            # for each graph-optimization level.
            optimized_graphs[level] = case_optimized_graph

    if original_graph is None:
        raise ValueError("No original ONNX graph was recorded")
    if set(optimized_graphs) != set(GRAPH_LEVELS):
        raise ValueError(
            "Expected one optimized t1 graph for each graph level"
        )

    operator_summary = {
        "schema_version": 1,
        "experiment": "day7_cpu_operator_profile_summary",
        "validation": {
            "record_count": len(records),
            "configuration_count": len(indexed),
            "operator_row_count": len(operators),
            "active_runs_per_profile": 10,
            "status": "passed",
        },
        "campaign": {
            "fingerprint": reference_fingerprint,
        },
        "profiles": profiles,
        "operators": operators,
    }

    graph_summary = {
        "schema_version": 1,
        "experiment": "day7_ort_optimized_graph_summary",
        "validation": {
            "original_graph_count": 1,
            "optimized_graph_count": len(optimized_graphs),
            "graph_optimization_levels": list(GRAPH_LEVELS),
            "status": "passed",
        },
        "original_graph": original_graph,
        "optimized_graphs": [
            compare_onnx_graph_summaries(
                level,
                original_graph,
                optimized_graphs[level],
            )
            for level in GRAPH_LEVELS
        ],
    }

    return (
        operator_summary,
        graph_summary,
        reference_fingerprint,
    )


def validate_cross_campaign(
    graph_optimization_summary: Mapping[str, Any],
    profile_fingerprint_value: Mapping[str, Any],
    graph_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Require formal latency and profiling evidence to share identity."""

    formal_fingerprint = field(
        graph_optimization_summary,
        "campaign.fingerprint",
    )

    # The two producer scripts have different source hashes by design. Compare
    # only the identity fields that prove a shared host, model, and input.
    shared_fields = (
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

    for name in shared_fields:
        if (
            formal_fingerprint[name]
            != profile_fingerprint_value[name]
        ):
            raise ValueError(
                f"Formal and profiling campaigns differ in {name}"
            )

    # The ONNX digest adds byte-level proof that both campaigns used the same
    # exported model rather than merely matching model metadata.
    original_graph = field(graph_summary, "original_graph")
    if (
        formal_fingerprint["onnx_sha256"]
        != original_graph["sha256"]
    ):
        raise ValueError(
            "Formal and profiling campaigns used different ONNX bytes"
        )

    return {
        "git_commit": formal_fingerprint["git_commit"],
        "weights_sha256": formal_fingerprint["weights_sha256"],
        "input_sha256": formal_fingerprint["input_sha256"],
        "onnx_sha256": formal_fingerprint["onnx_sha256"],
        "status": "passed",
    }


def profiling_operator_csv_rows(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return operator rows restricted to the stable CSV schema."""

    # The JSON summary retains richer backend-specific diagnostics; the CSV
    # projection exposes only the stable columns intended for comparison.
    return [
        {name: row.get(name) for name in OPERATOR_CSV_FIELDS}
        for row in summary["operators"]
    ]
