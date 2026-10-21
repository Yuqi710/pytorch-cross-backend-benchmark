"""Collect synchronized inference latency samples and persist statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import TYPE_CHECKING, Callable, Sequence, TypeVar

import numpy as np

# These imports provide static type infromation without loading the inference
# runtimes when this module is imported for statistics or unit testing.
if TYPE_CHECKING:
    import torch
    from inference_bench.backends import InferenceBackend

# Preserve each backend's native prepared-input and output types.
PreparedInput = TypeVar("PreparedInput")
NativeOutput = TypeVar("NativeOutput")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable iteration counts for warm-up and measured execution."""

    warmup: int = 50
    iterations: int = 300

    def __post_init__(self) -> None:
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")


@dataclass(frozen=True)
class LatencyMeasurement:
    """Raw latency samples and block-level timing for one benchmark run.

    samples_ns contains one synchronized inference latency per measured
    iteration. The block-level fields cover the complete post-warm-up
    sampling loop, including per-iteration boundary synchronization,
    timestamp collection, and output-reference release.

    average_process_cpu_core_equivalents is the ratio of accumulated
    process CPU time to elapsed wall time. It estimates average CPU
    consumption across the process and is not an active-thread count.
    """

    samples_ns: tuple[int, ...]
    sampling_block_wall_time_ns: int
    sampling_block_process_cpu_time_ns: int
    average_process_cpu_core_equivalents: float


@dataclass(frozen=True)
class LatencyStatistics:
    """Descriptive statistics for one latency sample sequence.

    Latency fields are expressed in milliseconds. Standard deviation uses
    ddof=0, and cv_percent expresses standard deviation relative to the mean.
    serial_service_rate_ips is derived from measured service time and excludeds
    overhead outside the measured intervals.
    """

    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    cv_percent: float
    serial_service_rate_ips: float


def summarize_latency(
    samples_ns: Sequence[int],
    *,
    batch_size: int = 1,
) -> LatencyStatistics:
    """Summarize positive integer nanoseconds in milliseconds.

    Percentiles uses Numpy's linear interpolation method. The caller must
    supply the batch size actually used to generate the samples.
    """

    if len(samples_ns) == 0:
        raise ValueError("At least one latency sample is required")
    if any(type(sample) is not int or sample <= 0 for sample in samples_ns):
        raise ValueError("Latency samples must be positive integer nanoseconds")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    # Keep raw samples as integer nanoseconds; use float64 for statistics.
    values_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000
    mean_ms = float(values_ms.mean())

    # Describe the observed sample sequence using the population denominator.
    std_ms = float(values_ms.std(ddof=0))
    p50, p90, p99 = np.percentile(
        values_ms, [50, 90, 99], method="linear"
    )

    # Convert Numpy scalar values to built-in floats for result serialization.
    return LatencyStatistics(
        count=len(samples_ns),
        mean_ms=mean_ms,
        p50_ms=float(p50),
        p90_ms=float(p90),
        p99_ms=float(p99),
        min_ms=float(values_ms.min()),
        max_ms=float(values_ms.max()),
        std_ms=std_ms,
        cv_percent=std_ms / mean_ms * 100,
        # Derived from summed measured service intervals, not a load test.
        serial_service_rate_ips=batch_size * 1000 / mean_ms,
    )


def collect_latency_measurement(
    backend: InferenceBackend[PreparedInput, NativeOutput],
    x: torch.Tensor,
    config: BenchmarkConfig,
    *,
    clock_ns: Callable[[], int] = perf_counter_ns,
    block_clock_ns: Callable[[], int] = perf_counter_ns,
    process_clock_ns: Callable[[], int] = process_time_ns,
) -> LatencyMeasurement:
    """Collect synchronized latency samples and block-level CPU usage.

    Input preparation and warm-up complete before either block-level
    clock starts. Each latency sample continues to include backend.run()
    and completion synchronizaiton while excluding input preparation,
    pre-sample synchronization, and explicit output-reference release.

    The outer wall-time interval covers the complete measured sampling
    loop. Process CPU time accumulates CPU work performed by the process
    during that interval and excludes time spent sleeping. Their ratio
    is reported as average CPU core equivalents rather than as a worker
    or active-thread count.

    Injectable clocks make the timing boundaries independently testable
    without changing the production clock implementations.
    """

    # Prepare input once and wait for preparation-related work to complete.
    prepared = backend.prepare_input(x)
    backend.synchronize()

    # Warm-up uses the same execution path without collecting timestamps.
    for _ in range(config.warmup):
        output = backend.run(prepared)
        backend.synchronize()
        del output

    # Start the wall clock before the process CPU clock.
    block_wall_start_ns = block_clock_ns()
    block_cpu_start_ns = process_clock_ns()

    samples: list[int] = []
    for _ in range(config.iterations):
        # Establish a clean boundary before starting the measured interval.
        backend.synchronize()
        start_ns = clock_ns()

        output = backend.run(prepared)

        # Include completion waiting so asynchronous execution is measured
        # through completion rather than only through task submission.
        backend.synchronize()
        end_ns = clock_ns()

        # Release this reference outside timing and before the next sample.
        # Backend allocators may retain the underlying storage in their caches.
        del output

        elapsed_ns = end_ns - start_ns
        if type(elapsed_ns) is not int or elapsed_ns <= 0:
            raise ValueError("Timer produced a nonpositive or invalid interval")
        samples.append(elapsed_ns)

    # Read the process CPU clock before the outer wall clock.
    block_cpu_time_ns = (
        process_clock_ns() - block_cpu_start_ns
    )
    block_wall_time_ns = (
        block_clock_ns() - block_wall_start_ns
    )

    if block_wall_time_ns <= 0:
        raise ValueError(
            "Sampling block wall time must be positive"
        )
    if block_cpu_time_ns < 0:
        raise ValueError(
            "Sampling block process CPU time cannot be negative"
        )

    return LatencyMeasurement(
        samples_ns=tuple(samples),
        sampling_block_wall_time_ns=block_wall_time_ns,
        sampling_block_process_cpu_time_ns=block_cpu_time_ns,
        # A value of 3.5 means that the process consumed CPU time at an
        # average rate equivalent to 3.5 fully utilized CPU cores.
        average_process_cpu_core_equivalents=(
            block_cpu_time_ns / block_wall_time_ns
        ),
    )


def collect_latency_samples(
    backend: InferenceBackend[PreparedInput, NativeOutput],
    x: torch.Tensor,
    config: BenchmarkConfig,
    *,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> tuple[int, ...]:
    """Return only raw latency samples for backward-compatible callers.

    New callers that also require process-level CPU observations should
    use collect_latency_measurement().
    """
    return collect_latency_measurement(
        backend,
        x,
        config,
        clock_ns=clock_ns,
    ).samples_ns


def save_json_result(result: dict[str, object], path: str | Path) -> Path:
    """Serialize strictly and create a new result file without overwriting.

    Serialization rejects non-finite floating-point values before the file
    is created. Exclusive creation prevents overwriting an existing result.
    This function does not validate the experiment's metadata schema or
    provide atomic whole-file replacement.
    """

    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Create exclusively; preserve any result already stored at this path.
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(content + "\n")
    return destination
