"""Benchmark one ONNX Runtime graph-optimization configuration.

Run one graph-optimization level and one explicit intra-op thread count in a
fresh process. Validate numerical correctness before and after synchronized
latency sampling, then preserve raw samples and reproducibility metadata.

This command collects formal latency evidence. It deliberately leaves runtime
profiling and optimized-model serialization disabled so their instrumentation
and file generation cannot affect the benchmark path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

from inference_bench.backends import OrtCpuBackend, run_once
from inference_bench.benchmark import (
    BenchmarkConfig,
    collect_latency_measurement,
    save_json_result,
    summarize_latency,
)
from inference_bench.models import load_model
from inference_bench.onnx_export import export_model_to_onnx
from inference_bench.ort_backend import GRAPH_OPTIMIZATION_LEVELS
from inference_bench.validation import compare_classification_outputs


ROOT = Path(__file__).resolve().parents[1]

# Keep the model contract, comparison tolerance, and export format fixed across
# every graph-optimization and thread configuration in the Day 7 campaign.
INPUT_SHAPE = (1, 3, 224, 224)
RTOL = ATOL = 1e-4
ONNX_OPSET_VERSION = 18


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


def file_hash(path: Path) -> str:
    """Return the SHA256 digest of a file's exact serialized contents."""

    digest = hashlib.sha256()
    # Read the ONNX artifact in bounded blocks rather than materializing the
    # complete serialized model as one additional byte string.
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash() -> str:
    """Fingerprint project Python sources and this benchmark entry point."""

    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_output(
    backend: OrtCpuBackend,
    x: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    """Validate one ORT output against the PyTorch CPU reference.

    Require both element-wise numerical closeness and Top-1 agreement. Raise
    before a successful result can be persisted when either condition fails.
    """

    # run_once() includes input preparation, execution, synchronization, and
    # conversion to an independent CPU tensor for correctness inspection.
    output = run_once(backend, x)
    comparison = compare_classification_outputs(
        reference, output, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(output, reference, rtol=RTOL, atol=ATOL)
    if not comparison.top1_match:
        raise AssertionError(f"{backend.name}: Top-1 mismatch")
    return asdict(comparison)


def parse_args() -> argparse.Namespace:
    """Parse one graph-optimization benchmark configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    # Derive CLI choices from the same stable labels used by session creation.
    parser.add_argument(
        "--graph-optimization-level",
        required=True,
        choices=list(GRAPH_OPTIMIZATION_LEVELS),
    )
    parser.add_argument(
        "--intra-op-threads",
        required=True,
        type=int,
        choices=[1, 6],
    )
    # The default counts match the established formal Day 5 and Day 6 latency
    # protocol. Shorter values remain available for local smoke verification.
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    # Repetition is implemented by fresh processes; this value labels one
    # member of the campaign rather than repeating work inside this process.
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/results/day7/staging/graph_optimization/raw"
        ),
    )

    args = parser.parse_args()

    if args.round < 1:
        parser.error("--round must be positive")

    return args


def main() -> None:
    """Run and persist one independent graph-optimization benchmark."""

    args = parse_args()

    # Validate iteration counts through the shared benchmark configuration so
    # Day 7 preserves the same warm-up and sampling contract as earlier days.
    config = BenchmarkConfig(warmup=args.warmup, iterations=args.iterations)

    # Combine a UTC timestamp with a random suffix so independent processes do
    # not reuse result filenames, even when they begin within the same second.
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]

    # Fix model and input identity across all optimization and thread cases.
    # The input uses its own generator so model construction cannot shift it.
    torch.manual_seed(2026)
    source = load_model("mobilenet_v3_small").cpu().eval()
    x = torch.randn(
        INPUT_SHAPE,
        generator=torch.Generator().manual_seed(2027),
        dtype=torch.float32,
    )

    # Persist independent fingerprints for model state and input values. The
    # campaign summarizer can reject results produced from mismatched inputs.
    weights_sha256 = model_hash(source)
    input_sha256 = hashlib.sha256(x.numpy().tobytes()).hexdigest()

    # Produce the reference before ONNX export from the exact in-memory model
    # whose state was fingerprinted above.
    with torch.inference_mode():
        reference = source(x).detach().clone()

    # Export happens before backend construction and remains outside all timed
    # intervals. Each process records the serialized graph hash so the formal
    # campaign can prove that every optimization level received the same graph.
    model_path = (
        ROOT / "artifacts/models" / "day7_mobilenet_v3_small_fp32.onnx"
    )
    export_model_to_onnx(
        copy.deepcopy(source),
        x,
        model_path,
        opset_version=ONNX_OPSET_VERSION,
    )
    onnx_model_sha256 = file_hash(model_path)

    # OrtCpuBackend supplies the same prepare/run/synchronize/to_cpu lifecycle
    # used by the shared benchmark sampler. It creates a regular session: the
    # requested graph optimizations execute during initialization, while
    # profiling and optimized-model serialization remain disabled.
    backend = OrtCpuBackend(
        model_path,
        intra_op_num_threads=args.intra_op_threads,
        graph_optimization_level=args.graph_optimization_level,
    )

    # Correctness gates surround sampling so invalid output cannot be reported
    # as a successful performance result and post-sampling behavior is checked.
    before = check_output(backend, x, reference)

    # The sampler prepares one backend-native input, performs untimed warm-up,
    # and records synchronized per-inference latency plus block-level CPU time.
    measurement = collect_latency_measurement(
        backend,
        x,
        config,
    )

    after = check_output(backend, x, reference)

    # Keep raw integer nanoseconds as primary evidence. Derived statistics are
    # stored for convenient inspection and can be recomputed from the samples.
    samples = measurement.samples_ns
    statistics = summarize_latency(samples, batch_size=1)
    backend_metadata = backend.metadata()
    clock = time.get_clock_info("perf_counter")

    result = {
        # Identify this record and the independent process that produced it.
        "schema_version": 1,
        "experiment": "day7_ort_graph_optimization",
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "round": args.round,
        "process_id": os.getpid(),
        "notes": args.notes,
        # Make the timed boundary and excluded setup work explicit alongside
        # the measurements so downstream interpretation does not rely on code.
        "measurement_scope": {
            "name": "prepared_input_synchronized_call",
            "timed_operations": [
                "backend.run",
                "backend.synchronize",
            ],
            "excluded_operations": [
                "model_loading",
                "onnx_export",
                "backend_initialization",
                "input_preparation",
                "correctness_checks",
                "warmup",
                "pre_sample_synchronization",
                "to_cpu",
                "explicit_output_reference_release",
                "result_serialization",
            ],
            "request_concurrency": 1,
            "thread_policy": "explicit_intra_op_inter_op_runtime_default",
            "graph_optimization_policy": "one_level_per_independent_process",
            "input_policy": "same_prepared_synthetic_input_reused",
            "percentile_method": "numpy_linear",
            "std_ddof": 0,
            "serial_service_rate_definition": (
                "batch_size * 1e9 / mean(samples_ns)"
            ),
        },
        # Record both the requested thread policy and the runtime-reported
        # session values. These values are configuration, not worker counts.
        "threading": {
            "experiment_variable": "intra_op_threads",
            "intra_op": {
                "policy": "explicit",
                "requested_threads": args.intra_op_threads,
                "runtime_reported_value": backend_metadata[
                    "intra_op_num_threads"
                ],
                "scope": "onnxruntime_cpu_operator_intra_op",
            },
            "inter_op": {
                "policy": "runtime_default",
                "requested_threads": None,
                "runtime_reported_value": backend_metadata[
                    "inter_op_num_threads"
                ],
            },
        },
        # Preserve the stable experiment label and ORT's native enum string so
        # the summarizer can verify that the requested policy was applied.
        "graph_optimization": {
            "experiment_variable": "graph_optimization_level",
            "requested_level": args.graph_optimization_level,
            "runtime_level_name": backend_metadata[
                "graph_optimization_level_name"
            ],
            "runtime_level": backend_metadata["graph_optimization_level"],
            "execution_mode": backend_metadata["execution_mode"],
        },
        # Formal latency sessions exclude the two diagnostic features exercised
        # separately by day7_profile_cpu.py.
        "diagnostic_features": {
            "profiling_enabled": False,
            "optimized_model_serialization_enabled": False,
        },
        # Snapshot host, runtime, clock, and inherited thread-pool state after
        # inference collection for reproducibility and campaign validation.
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
            "clock": {
                "name": "perf_counter_ns",
                "implementation": clock.implementation,
                "resolution_seconds": clock.resolution,
                "monotonic": clock.monotonic,
            },
        },
        # Process CPU time divided by wall time estimates average CPU core
        # equivalents; it must not be interpreted as an active-thread count.
        "cpu_process_observation": {
            "scope": "complete_post_warmup_sampling_loop",
            "sampling_block_wall_time_ns": (
                measurement.sampling_block_wall_time_ns
            ),
            "sampling_block_process_cpu_time_ns": (
                measurement.sampling_block_process_cpu_time_ns
            ),
            "average_process_cpu_core_equivalents": (
                measurement.average_process_cpu_core_equivalents
            ),
            "interpretation": (
                "aggregate_process_cpu_usage_not_active_thread_count"
            ),
        },
        # Record committed identity together with current source contents so a
        # result remains auditable even if a smoke run uses local changes.
        "source": {
            "git_commit": read_command(["git", "rev-parse", "HEAD"]),
            "git_status": read_command(["git", "status", "--porcelain"]),
            "source_sha256": source_hash(),
        },
        # Bind the result to exact PyTorch state, input values, and serialized
        # ONNX bytes rather than relying on model names alone.
        "model": {
            "name": "mobilenet_v3_small",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "weights_sha256": weights_sha256,
            "input_sha256": input_sha256,
            "input_shape": list(INPUT_SHAPE),
            "precision": "fp32",
            "model_seed": 2026,
            "input_seed": 2027,
            "onnx": {
                "path": model_path.relative_to(ROOT).as_posix(),
                "opset_version": ONNX_OPSET_VERSION,
                "sha256": onnx_model_sha256,
                "file_size_bytes": model_path.stat().st_size,
            },
        },
        "backend": backend_metadata,
        # Store both correctness gates around the measured sampling block.
        "correctness": {
            "before": before,
            "after": after,
        },
        "benchmark": asdict(config),
        # Preserve every observed latency as integer nanoseconds for independent
        # aggregation and alternative statistical analysis.
        "samples_ns": list(samples),
        "statistics": asdict(statistics),
        "status": "passed",
    }

    # Resolve the default artifact location from the repository root so command
    # behavior is independent of the caller's current working directory.
    output_dir = args.output_dir

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    # Encode both experiment variables and the round in the filename; the run
    # identifier keeps artifacts from independent processes immutable.
    path = output_dir / (
        "ort_cpu_"
        f"opt-{args.graph_optimization_level}_"
        f"t{args.intra_op_threads}_"
        f"r{args.round}_"
        f"{run_id}.json"
    )

    # save_json_result() uses exclusive creation and therefore refuses to
    # overwrite an existing raw result.
    save_json_result(result, path)

    print(
        "ort_cpu "
        f"graph_opt={args.graph_optimization_level} "
        f"intra_op={args.intra_op_threads} "
        f"round={args.round} "
        f"n={statistics.count} "
        f"p50={statistics.p50_ms:.4f}ms "
        f"p90={statistics.p90_ms:.4f}ms "
        f"p99={statistics.p99_ms:.4f}ms "
        f"mean={statistics.mean_ms:.4f}ms "
        f"CV={statistics.cv_percent:.2f}% "
        f"derived_serial_rate={statistics.serial_service_rate_ips:.2f} images/s"
    )
    print(f"Result saved to: {path}")


if __name__ == "__main__":
    main()