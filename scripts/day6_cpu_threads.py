"""Benchmark one backend and intra-op thread configuration per process.

Validate numerical correctness before and after latency sampling, and persist
raw samples with runtime configuration and reproducibility metadata.
"""

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

from inference_bench.backends import OrtCpuBackend, TorchBackend, run_once
from inference_bench.benchmark import (
    BenchmarkConfig,
    collect_latency_measurement,
    save_json_result,
    summarize_latency,
)
from inference_bench.models import load_model
from inference_bench.onnx_export import export_model_to_onnx
from inference_bench.validation import compare_classification_outputs


ROOT = Path(__file__).resolve().parents[1]

# Keep shape, precision, and tolerances consistent across backend experiments.
INPUT_SHAPE = (1, 3, 224, 224)
RTOL = ATOL = 1e-4


def read_command(args):
    """Return command stdout, or None when metadata cannot be collected.

    Commands run without a shell and outside the latency sampling interval.
    An unavailable metadata command does not invalidate inference results.
    """
    try:
        result = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def model_hash(model):
    """Fingerprint model parameters and buffers using a stable key order.

    Include tensor names, dtypes, shapes, and values so that experiments can
    distinguish models with different state despite matching model labels.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hash():
    """Fingerprint project Python sources and this benchmark entry point.

    Relative paths participate in the hash. Tests, shell scripts, dependency
    files, and generated artifacts are outside this fingerprint's scope.
    """
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_output(backend, x, reference):
    """Validate one backend execution against the PyTorch CPU reference.

    Require both element-wise numerical closeness and Top-1 agreement.
    Raise on failure before a successful benchmark result can be persisted.
    """
    output = run_once(backend, x)
    comparison = compare_classification_outputs(
        reference, output, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(output, reference, rtol=RTOL, atol=ATOL)
    if not comparison.top1_match:
        raise AssertionError(f"{backend.name}: Top-1 mismatch")
    return asdict(comparison)


def main():
    """Run one experiment for the requested backend and round label."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", required=True,
        choices=["pytorch_cpu", "pytorch_mps", "ort_cpu"],
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--intra-op-threads", type=int, default=None)
    # The shell runner controls repetition; this argument labels one run.
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/day6/staging/cpu_threads/raw"),
    )
    args = parser.parse_args()

    config = BenchmarkConfig(warmup=args.warmup, iterations=args.iterations)

    if args.round < 1:
        parser.error("--round must be positive")

    # Reject invalid thread counts before constructing models or runtimes.
    if args.intra_op_threads is not None and args.intra_op_threads < 1:
        parser.error("--intra-op-threads must be positive")

    # Fail explicitly when the requested MPS experiment cannot be executed.
    if args.backend == "pytorch_mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS is unavailable; requested experiment cannot run")
        # Formal MPS runs require environment-controlled CPU fallback to be
        # explicitly disabled before the process starts.
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            parser.error(
                "Day6 MPS runs require "
                "PYTORCH_ENABLE_MPS_FALLBACK=0"
            )

    # Apply PyTorch's process-wide intra-op setting before model construction,
    # reference inference, or backend initialization. Inter-op configuration
    # remains at the runtime default.
    if (
        args.backend in {"pytorch_cpu", "pytorch_mps"}
        and args.intra_op_threads is not None
    ):
        torch.set_num_threads(args.intra_op_threads)

    # Timestamp and random suffix identify independent runs without reusing
    # result filenames.
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]

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

    # Instantiate only the selected benchmark backend.
    if args.backend == "ort_cpu":
        model_path = ROOT / "artifacts/models/day6_mobilenet_v3_small_fp32.onnx"
        export_model_to_onnx(copy.deepcopy(source), x, model_path, opset_version=18)
        # Apply the requested count only to the ORT intra-op thread pool.
        # Sequential graph scheduling and the runtime-default inter-op policy
        # remain unchanged.
        backend = OrtCpuBackend(
            model_path,
            intra_op_num_threads=args.intra_op_threads,
        )
    else:
        device = "mps" if args.backend == "pytorch_mps" else "cpu"
        backend = TorchBackend(source, device=device)

    # Correctness checks surround sampling. The sampler owns input preparation,
    # warm-up, per-sample execution, and completion synchronization.
    before = check_output(backend, x, reference)
    # Collect synchronized per-inference latency samples together with aggregate
    # process CPU time over the complete post-warm-up sampling block.
    measurement = collect_latency_measurement(backend, x, config)
    samples = measurement.samples_ns
    after = check_output(backend, x, reference)

    statistics = summarize_latency(samples, batch_size=1)
    clock = time.get_clock_info("perf_counter")
    # Snapshot backend-native metadata once so the raw backend record and the
    # normalized threading record describe the same runtime state.
    backend_metadata = backend.metadata()

    # Normalize backend-specific thread field names for cross-backend aggregation.
    # The original runtime-specific fields remain available in backend_metadata.
    if args.backend == "ort_cpu":
        runtime_intra_op = backend_metadata["intra_op_num_threads"]
        runtime_inter_op = backend_metadata["inter_op_num_threads"]
        intra_op_scope = "onnxruntime_cpu_operator_intra_op"
    else:
        runtime_intra_op = backend_metadata["torch_num_threads"]
        runtime_inter_op = backend_metadata["torch_num_interop_threads"]
        # For MPS, the reported PyTorch value describes the host CPU intra-op setting;
        # it does not represent an MPS worker count or measured GPU parallelism.
        intra_op_scope = (
            "pytorch_cpu_intra_op_setting_not_mps_worker_count"
            if args.backend == "pytorch_mps"
            else "pytorch_cpu_operator_intra_op"
        )

    # Preserve both the requested policy and the runtime-reported value so that a
    # default configuration remains distinguishable from an explicit request with
    # the same numeric value.
    intra_op_metadata = {
        "policy": (
            "runtime_default"
            if args.intra_op_threads is None
            else "explicit"
        ),
        "requested_threads": args.intra_op_threads,
        "runtime_reported_value": runtime_intra_op,
        "scope": intra_op_scope,
    }
    if args.backend == "ort_cpu" and args.intra_op_threads is None:
        intra_op_metadata["runtime_reported_value_semantics"] = (
            "runtime_default_sentinel"
        )

    result = {
        "schema_version": 1,
        "experiment": "day6_cpu_intra_op_thread_sweep",
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "round": args.round,
        "process_id": os.getpid(),
        "notes": args.notes,
        # Describe the measurement boundary alongside the numeric results.
        "measurement_scope": {
            "name": "prepared_input_synchronized_call",
            "timed_operations": ["backend.run", "backend.synchronize"],
            "excluded_operations": [
                "model_loading", "onnx_export", "backend_initialization",
                "input_preparation", "correctness_checks", "warmup",
                "pre_sample_synchronization", "to_cpu",
                "explicit_output_reference_release", "result_serialization",
            ],
            "request_concurrency": 1,
            "thread_policy": "intra_op_swept_inter_op_runtime_default",
            "input_policy": "same_prepared_synthetic_input_reused",
            "percentile_method": "numpy_linear",
            "std_ddof": 0,
            # This is derived from measured service intervals, rather than
            # observed throughput under concurrent request load.
            "serial_service_rate_definition": "batch_size * 1e9 / mean(samples_ns)",
        },
        # Store a backend-independent threading view for automated comparison while
        # retaining the backend-native configuration for auditing.
        "threading": {
            "experiment_variable": "intra_op_threads",
            "backend_role": (
                "mps_host_thread_sensitivity_control"
                if args.backend == "pytorch_mps"
                else "primary_cpu_scan"
            ),
            "intra_op": intra_op_metadata,
            "inter_op": {
                "policy": "runtime_default",
                "requested_threads": None,
                "runtime_reported_value": runtime_inter_op,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "system": platform.system(),
            "macos_version": platform.mac_ver()[0],
            "logical_cpu_count": os.cpu_count(),
            "cpu_model": read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "hardware_model": read_command(["sysctl", "-n", "hw.model"]),
            "memory_bytes": read_command(["sysctl", "-n", "hw.memsize"]),
            "power_source_snapshot_after_run": read_command(["pmset", "-g", "batt"]),
            # Record inherited thread-pool environment settings even though the formal
            # campaign requires them to be unset.
            "thread_environment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get(
                    "VECLIB_MAXIMUM_THREADS"
                ),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
            "versions": {
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                "numpy": np.__version__,
            },
            "clock": {
                "name": "perf_counter_ns", "implementation": clock.implementation,
                "resolution_seconds": clock.resolution, "monotonic": clock.monotonic,
            },
        },
        # Report block-level process CPU consumption separately from per-call latency.
        # The CPU-time-to-wall-time ratio estimates average core-equivalent usage and
        # must not be interpreted as an active-thread count.
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
        # Record both committed identity and current source contents because
        # experiments may be executed with uncommitted changes.
        "source": {
            "git_commit": read_command(["git", "rev-parse", "HEAD"]),
            "git_status": read_command(["git", "status", "--porcelain"]),
            "source_sha256": source_hash(),
        },
        "model": {
            "name": "mobilenet_v3_small",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "weights_sha256": weights_sha256, "input_sha256": input_sha256,
            "input_shape": list(INPUT_SHAPE), "precision": "fp32",
            "model_seed": 2026, "input_seed": 2027,
        },
        "backend": backend_metadata,
        "correctness": {"before": before, "after": after},
        "benchmark": asdict(config),
        # Preserve integer nanosecond samples for later statistical analysis.
        "samples_ns": list(samples),
        "statistics": asdict(statistics),
        "status": "passed",
    }
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    thread_label = (
        "default"
        if args.intra_op_threads is None
        else str(args.intra_op_threads)
    )
    path = output_dir / (
        f"{args.backend}_t{thread_label}_r{args.round}_{run_id}.json"
    )
    save_json_result(result, path)
    print(
        f"{args.backend} intra_op={thread_label} "
        f"round={args.round} n={statistics.count} "
        f"p50={statistics.p50_ms:.4f}ms p90={statistics.p90_ms:.4f}ms "
        f"p99={statistics.p99_ms:.4f}ms mean={statistics.mean_ms:.4f}ms "
        f"CV={statistics.cv_percent:.2f}% "
        f"derived_serial_rate={statistics.serial_service_rate_ips:.2f} images/s"
    )
    print(f"Result saved to: {path}")


if __name__ == "__main__":
    main()
