"""Collect one Day7 CPU operator-profiling result.

This command profiles one backend and one explicit intra-op thread setting in
a fresh process. PyTorch Profiler and ONNX Runtime Profiling are diagnostic
tools, so their instrumented timings are stored separately from formal latency
measurements.

The command validates numerical correctness before and after profiling,
preserves the raw trace, and write a compact result containing operator,
runtime, graph, environment, and source metadata.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

from inference_bench.backends import TorchBackend, run_once
from inference_bench.benchmark import save_json_result
from inference_bench.models import load_model
from inference_bench.onnx_export import export_model_to_onnx
from inference_bench.ort_backend import (
    GRAPH_OPTIMIZATION_LEVELS,
    GraphOptimizationLevelName,
    create_cpu_session,
    run_ort_inference,
)
from inference_bench.profiling import (
    file_sha256,
    summarize_onnx_graph,
    summarize_ort_profile,
    summarize_torch_profile,
)
from inference_bench.validation import compare_classification_outputs


ROOT = Path(__file__).resolve().parents[1]
INPUT_SHAPE = (1, 3, 224, 224)
RTOL = ATOL = 1e-4

# PyTorch uses a scheduled capture so profiler startup and warm-up activity do
# not contribute to the operator summary.
PYTORCH_PROFILE_WAIT_STEPS = 1
PYTORCH_PROFILE_WARMUP_STEPS = 2


def read_command(args: list[str]) -> str | None:
    """Return command stdout, or None when metadata cannot be collected."""

    try:
        result = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def model_hash(model: torch.nn.Module) -> str:
    """Fingerprint model parameters and buffers in stable key order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    """Fingerprint project Python sources and this profiling script."""

    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def project_path(path: Path) -> Path:
    """Resolve a command-line path relative to the repository root."""

    return path if path.is_absolute() else ROOT / path


def recorded_path(path: Path) -> str:
    """Return a portable path for persisted experiment metadata."""

    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        return resolved.relative_to(ROOT).as_posix()
    return resolved.as_posix()


def graph_summary(path: Path) -> dict[str, object]:
    """Summarize an ONNX graph and normalize its recorded location."""

    summary = summarize_onnx_graph(path)
    summary["path"] = recorded_path(path)
    return summary


def check_output(
    backend_name: str,
    output: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    """Require numerical closeness and Top-1 agreement."""

    comparison = compare_classification_outputs(
        reference, output, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(output, reference, rtol=RTOL, atol=ATOL)
    if not comparison.top1_match:
        raise AssertionError(f"{backend_name}: Top-1 mismatch")
    return asdict(comparison)


def profile_pytorch(
    backend: TorchBackend,
    x: torch.Tensor,
    *,
    external_warmup_runs: int,
    active_runs: int,
    trace_path: Path,
) -> dict[str, object]:
    """Capture the scheduled active window of a PyTorch CPU profile.

    Input preparation and external model warm-up complete before profiler
    collection begins. The profiler schedule then separates its own wait,
    warm-up, and active phases. Only the active phase contributes to the
    exported operator summary.
    """

    prepared = backend.prepare_input(x)

    # Stabilize model execution and allocator state before starting profiler
    # instrumentation. TorchBackend.run() already executes in inference mode.
    for _ in range(external_warmup_runs):
        output = backend.run(prepared)
        del output

    schedule = torch.profiler.schedule(
        wait=PYTORCH_PROFILE_WAIT_STEPS,
        warmup=PYTORCH_PROFILE_WARMUP_STEPS,
        active=active_runs,
        repeat=1,
    )
    scheduled_runs = (
        PYTORCH_PROFILE_WAIT_STEPS
        + PYTORCH_PROFILE_WARMUP_STEPS
        + active_runs
    )

    # Shape and memory collection intentionally remain limited to diagnostic
    # runs because both features add instrumentation overhead.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(scheduled_runs):
            output = backend.run(prepared)
            del output
            profiler.step()

    profiler.export_chrome_trace(str(trace_path))
    return summarize_torch_profile(profiler)


def profile_ort(
    session: ort.InferenceSession,
    x: torch.Tensor,
    *,
    external_warmup_runs: int,
    active_runs: int,
    trace_path: Path,
) -> dict[str, object]:
    """Capture and summarize the final active runs of an ORT profile.

    ORT profiling starts when the session is initialized, so the raw trace also
    contains session setup, the pre-profile correctness call, and external
    warm-up. Profiling ends immediately after the active calls, allowing the
    summary parser to identify them as the final ``active_runs`` model-run
    intervals in the trace.
    """

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_array = x.detach().cpu().contiguous().numpy()

    # Reuse one prepared NumPy input so conversion and allocation remain
    # outside the inference calls selected for the operator summary.
    for _ in range(external_warmup_runs):
        output = session.run([output_name], {input_name: input_array})
        del output

    for _ in range(active_runs):
        output = session.run([output_name], {input_name: input_array})
        del output

    # ORT chooses the generated filename from the configured prefix. Rename it
    # to the run-specific artifact name shared by the remaining result files.
    generated_trace = Path(session.end_profiling())
    generated_trace.rename(trace_path)

    return summarize_ort_profile(trace_path, active_runs=active_runs)


def ort_session_metadata(
    session: ort.InferenceSession,
    graph_optimization_level: GraphOptimizationLevelName,
) -> dict[str, object]:
    """Report the effective ORT provider, graph, and threading settings."""

    options = session.get_session_options()
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]

    return {
        "name": "ort_cpu",
        "runtime": "onnxruntime",
        "runtime_version": ort.__version__,
        "device": "cpu",
        "precision": "fp32",
        "input_shape": list(model_input.shape),
        "input_name": model_input.name,
        "output_name": model_output.name,
        "providers": session.get_providers(),
        "graph_optimization_level_name": graph_optimization_level,
        "graph_optimization_level": str(options.graph_optimization_level),
        "execution_mode": str(options.execution_mode),
        "intra_op_num_threads": options.intra_op_num_threads,
        "inter_op_num_threads": options.inter_op_num_threads,
        "profiling_enabled": options.enable_profiling,
    }


def parse_args() -> argparse.Namespace:
    """Parse and validate one profiling configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", required=True, choices=["pytorch_cpu", "ort_cpu"]
    )
    parser.add_argument(
        "--intra-op-threads", required=True, type=int, choices=[1, 6]
    )
    parser.add_argument(
        "--graph-optimization-level",
        choices=list(GRAPH_OPTIMIZATION_LEVELS),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--active-runs", type=int, default=10)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/day7/staging/profiling/raw"),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("artifacts/results/day7/local_archive/traces"),
    )
    parser.add_argument("--optimized-model-path", type=Path, default=None)

    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")

    if args.active_runs < 1:
        parser.error("--active-runs must be positive")

    if args.backend == "pytorch_cpu" and args.optimized_model_path is not None:
        parser.error("--optimized-model-path is available only for ort_cpu")

    return args


def main() -> None:
    """Run and persist one independent profiling configuration."""

    args = parse_args()

    # PyTorch intra-op state is process-wide and must be configured before
    # model construction, reference execution, or backend initialization.
    if args.backend == "pytorch_cpu":
        torch.set_num_threads(args.intra_op_threads)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]

    # Use fixed model and input identities across every Day 7 configuration.
    torch.manual_seed(2026)
    source = load_model("mobilenet_v3_small").cpu().eval()
    x = torch.randn(
        INPUT_SHAPE,
        generator=torch.Generator().manual_seed(2027),
        dtype=torch.float32,
    )

    weights_sha256 = model_hash(source)
    input_sha256 = hashlib.sha256(x.numpy().tobytes()).hexdigest()

    with torch.inference_mode():
        reference = source(x).detach().clone()

    output_dir = project_path(args.output_dir)
    trace_dir = project_path(args.trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # A timestamp and random suffix give every independent process its own
    # immutable trace and compact result record.
    if args.backend == "pytorch_cpu":
        file_stem = f"pytorch_cpu_t{args.intra_op_threads}_{run_id}"
    else:
        file_stem = (
            f"ort_cpu_opt-{args.graph_optimization_level}_"
            f"t{args.intra_op_threads}_{run_id}"
        )

    trace_path = trace_dir / f"{file_stem}.trace.json"
    result_path = output_dir / f"{file_stem}.json"

    if trace_path.exists():
        raise FileExistsError(trace_path)

    optimized_model_path = (
        project_path(args.optimized_model_path)
        if args.optimized_model_path is not None
        else None
    )

    if optimized_model_path is not None:
        optimized_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Optimized graphs are raw experiment artifacts and must not replace a
        # graph produced by an earlier run.
        if optimized_model_path.exists():
            raise FileExistsError(optimized_model_path)

    original_onnx_graph = None
    optimized_graph = None

    if args.backend == "pytorch_cpu":
        backend = TorchBackend(source, device="cpu")

        # Correctness checks surround profiling while remaining outside the
        # captured active window.
        before = check_output(
            backend.name,
            run_once(backend, x),
            reference,
        )
        profile_summary = profile_pytorch(
            backend,
            x,
            external_warmup_runs=args.warmup,
            active_runs=args.active_runs,
            trace_path=trace_path,
        )
        after = check_output(
            backend.name,
            run_once(backend, x),
            reference,
        )

        backend_metadata = backend.metadata()
        graph_optimization_level = None
        runtime_intra_op_threads = backend_metadata["torch_num_threads"]
        runtime_inter_op_threads = backend_metadata[
            "torch_num_interop_threads"
        ]

    else:
        model_path = (
            ROOT
            / "artifacts/models"
            / "day7_mobilenet_v3_small_fp32.onnx"
        )

        # Export a copy because the export helper may change device placement
        # and evaluation state on the module it receives.
        export_model_to_onnx(
            copy.deepcopy(source),
            x,
            model_path,
            opset_version=18,
        )
        original_onnx_graph = graph_summary(model_path)

        profile_prefix = trace_dir / f"{file_stem}.ort"
        session = create_cpu_session(
            model_path,
            intra_op_num_threads=args.intra_op_threads,
            graph_optimization_level=args.graph_optimization_level,
            profile_file_prefix=profile_prefix,
            optimized_model_path=optimized_model_path,
        )

        # This inference is included in the raw ORT trace but precedes
        # warm-up and active collection. summarize_ort_profile() selects
        # only the final active_runs model_run intervals.
        before = check_output(
            "ort_cpu",
            run_ort_inference(session, x),
            reference,
        )
        profile_summary = profile_ort(
            session,
            x,
            external_warmup_runs=args.warmup,
            active_runs=args.active_runs,
            trace_path=trace_path,
        )

        # end_profiling() has closed the trace, so the post-profile validation
        # cannot enter either the raw trace or the active operator summary.
        after = check_output(
            "ort_cpu",
            run_ort_inference(session, x),
            reference,
        )

        backend_metadata = ort_session_metadata(
            session,
            args.graph_optimization_level,
        )
        graph_optimization_level = args.graph_optimization_level
        runtime_intra_op_threads = backend_metadata["intra_op_num_threads"]
        runtime_inter_op_threads = backend_metadata["inter_op_num_threads"]

        if optimized_model_path is not None:
            optimized_graph = graph_summary(optimized_model_path)

    # Profiler durations describe instrumented diagnostic execution. The result
    # therefore records profiling scope explicitly and does not expose these
    # values as formal inference latency measurements.
    result = {
        "schema_version": 1,
        "experiment": "day7_cpu_operator_profile",
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "process_id": os.getpid(),
        "notes": args.notes,
        "profiling_scope": {
            "purpose": "operator_diagnostics_not_formal_latency",
            "summary_window": "active_inference_runs_only",
            "request_concurrency": 1,
            "input_policy": "same_prepared_synthetic_input_reused",
        },
        "profile_config": {
            "external_warmup_runs": args.warmup,
            "active_runs": args.active_runs,
            "pytorch_schedule": (
                {
                    "wait_steps": PYTORCH_PROFILE_WAIT_STEPS,
                    "warmup_steps": PYTORCH_PROFILE_WARMUP_STEPS,
                    "active_steps": args.active_runs,
                    "repeat": 1,
                }
                if args.backend == "pytorch_cpu"
                else None
            ),
            "record_shapes": args.backend == "pytorch_cpu",
            "profile_memory": args.backend == "pytorch_cpu",
            "with_stack": False,
        },
        "threading": {
            "intra_op": {
                "policy": "explicit",
                "requested_threads": args.intra_op_threads,
                "runtime_reported_value": runtime_intra_op_threads,
            },
            "inter_op": {
                "policy": "runtime_default",
                "requested_threads": None,
                "runtime_reported_value": runtime_inter_op_threads,
            },
        },
        "graph_optimization_level": graph_optimization_level,
        "environment": {
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "system": platform.system(),
            "macos_version": platform.mac_ver()[0],
            "logical_cpu_count": os.cpu_count(),
            "cpu_model": read_command(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ),
            "hardware_model": read_command(["sysctl", "-n", "hw.model"]),
            "memory_bytes": read_command(["sysctl", "-n", "hw.memsize"]),
            "power_source_snapshot_after_run": read_command(
                ["pmset", "-g", "batt"]
            ),
            "thread_environment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get(
                    "OPENBLAS_NUM_THREADS"
                ),
                "VECLIB_MAXIMUM_THREADS": os.environ.get(
                    "VECLIB_MAXIMUM_THREADS"
                ),
                "NUMEXPR_NUM_THREADS": os.environ.get(
                    "NUMEXPR_NUM_THREADS"
                ),
            },
            "versions": {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": ort.__version__,
                "numpy": np.__version__,
            },
        },
        "source": {
            "git_commit": read_command(["git", "rev-parse", "HEAD"]),
            "git_status": read_command(["git", "status", "--porcelain"]),
            "source_sha256": source_hash(),
        },
        "model": {
            "name": "mobilenet_v3_small",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "weights_sha256": weights_sha256,
            "input_sha256": input_sha256,
            "input_shape": list(INPUT_SHAPE),
            "precision": "fp32",
            "model_seed": 2026,
            "input_seed": 2027,
            "original_onnx_graph": original_onnx_graph,
        },
        "backend": backend_metadata,
        "correctness": {
            "before": before,
            "after": after,
        },
        "trace": {
            "path": recorded_path(trace_path),
            "sha256": file_sha256(trace_path),
        },
        "optimized_graph": optimized_graph,
        "profile_summary": profile_summary,
        "status": "passed",
    }

    save_json_result(result, result_path)

    print(
        f"{args.backend} "
        f"intra_op={args.intra_op_threads} "
        f"graph_opt={graph_optimization_level} "
        f"active_runs={args.active_runs}"
    )
    print(f"Trace saved to: {recorded_path(trace_path)}")
    print(f"Result saved to: {recorded_path(result_path)}")


if __name__ == "__main__":
    main()