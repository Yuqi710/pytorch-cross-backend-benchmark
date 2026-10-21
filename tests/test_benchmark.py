"""Unit test for benchmark configuration, statistics, sampling, and persistence."""

import json
import math

import pytest

from inference_bench import benchmark


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup": -1},
        {"warmup": 1.5},
        {"warmup": True},
        {"iterations": 0},
        {"iterations": -1},
        {"iterations": 2.5},
        {"iterations": True},
    ],
)
def test_config_rejects_invalid_counts(kwargs):
    """Require valid integer counts for warm-up and measured iterations."""
    # Booleans must be rejected even though bool is a subclass of int.
    with pytest.raises(ValueError):
        benchmark.BenchmarkConfig(**kwargs)


def test_summary_uses_ms_linear_percentiles_and_serial_service_rate():
    """Verify unit conversion and independently calculated statistics."""
    # These integer nanosecond samples correspond to [1, 2, 3, 4] ms.
    result = benchmark.summarize_latency(
        [1_000_000, 2_000_000, 3_000_000, 4_000_000],
        batch_size=1,
    )
    assert result.count == 4
    assert result.mean_ms == pytest.approx(2.5)

    # Linear interpolation uses the index (n - 1) * quantile.
    assert result.p50_ms == pytest.approx(2.5)
    assert result.p90_ms == pytest.approx(3.7)
    assert result.p99_ms == pytest.approx(3.97)

    assert result.min_ms == pytest.approx(1.0)
    assert result.max_ms == pytest.approx(4.0)

    # Population variance is 1.25 ms^2; standard deviation uses ddof=0.
    assert result.std_ms == pytest.approx(math.sqrt(1.25))
    assert result.cv_percent == pytest.approx(math.sqrt(1.25) / 2.5 * 100)

    # One image per 2.5 ms gives a derived serial service rate of 400 image/s.
    assert result.serial_service_rate_ips == pytest.approx(400.0)


@pytest.mark.parametrize("samples", [[], [0], [-1], [1.2], [True]])
def test_summary_rejects_invalid_samples(samples):
    """Require a nonempty sequence of positive integer nanosecond samples."""
    with pytest.raises(ValueError):
        benchmark.summarize_latency(samples)


@pytest.mark.parametrize("batch_size", [0, True, 1.5])
def test_summary_rejects_invalid_batch_size(batch_size):
    """Require a positive integer batch size for service-rate calculation."""
    with pytest.raises(ValueError):
        benchmark.summarize_latency([1_000_000], batch_size=batch_size)


class RecordingBackend:
    """Test double for timing orchestration; no hardware timing claims."""

    def __init__(self, events):
        self.events = events
        self.prepared = None

    def prepare_input(self, x):
        self.events.append("prepare")
        self.prepared = {"input": x}
        return self.prepared

    def run(self, prepared):
        # Every invocation must reuse the object returned by prepare_input().
        assert prepared is self.prepared
        self.events.append("run")
        return object()

    def synchronize(self):
        self.events.append("sync")

    def to_cpu(self, output):
        # Output conversion belongs to the caller, outside the sampling routine.
        raise AssertionError("Output conversion entered timing sampler")


def test_sampler_excludes_preparation_warmup_and_conversion():
    """Verify sample durations and synchronization relative to timestamps."""
    events = []
    backend = RecordingBackend(events)

    # Two measured intervals: 3 ms and 5 ms. The gap is excluded.
    ticks = iter([0, 3_000_000, 9_000_000, 14_000_000])
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        events.append("start" if clock_calls % 2 == 0 else "end")
        clock_calls += 1
        return next(ticks)

    # The injected clock makes timing behavior deterministic.
    samples = benchmark.collect_latency_samples(
        backend,
        object(),
        benchmark.BenchmarkConfig(warmup=1, iterations=2),
        clock_ns=clock,
    )

    assert samples == (3_000_000, 5_000_000)
    assert events == [
        # Input preparation and one untimed warm-up execution.
        "prepare", "sync", "run", "sync",
        # Synchronize before starting; include completion synchronization.
        "sync", "start", "run", "sync", "end",
        "sync", "start", "run", "sync", "end",
    ]


@pytest.mark.parametrize("ticks", [(1, 1), (2, 1)])
def test_sampler_rejects_nonpositive_clock_intervals(ticks):
    """Reject zero or negative elapsed time instead of recording invalid data."""
    clock = iter(ticks)
    with pytest.raises(ValueError):
        benchmark.collect_latency_samples(
            RecordingBackend([]),
            object(),
            benchmark.BenchmarkConfig(warmup=0, iterations=1),
            clock_ns=lambda: next(clock),
        )


def test_sampler_propagates_execution_failure():
    """Ensure backend execution failures remain visible to the caller."""
    class FailingBackend(RecordingBackend):
        def run(self, prepared):
            raise RuntimeError("execution failed")

    with pytest.raises(RuntimeError, match="execution failed"):
        benchmark.collect_latency_samples(
            FailingBackend([]),
            object(),
            benchmark.BenchmarkConfig(warmup=0, iterations=1),
        )


def test_json_result_preserves_metadata_and_raw_integer_samples(tmp_path):
    """Varify directory creation and a lossless JSON round trip."""
    result = {
        "schema_version": 1,
        "backend": {"name": "ort_cpu", "threads": 0},
        "samples_ns": [1_000_000, 2_000_000],
    }
    path = tmp_path / "nested" / "result.json"
    benchmark.save_json_result(result, path)
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_json_result_refuses_overwrite(tmp_path):
    """Preserve existing experiment results when a destination already exists."""
    path = tmp_path / "result.json"
    path.write_text('{"original": true}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark.save_json_result({"replacement": True}, path)
    # A failed write must leave the existing content unchanged.
    assert json.loads(path.read_text(encoding="utf-8")) == {"original": True}


def test_json_result_rejects_nonfinite_values_before_creating_file(tmp_path):
    """Reject NaN during serialization before creating the destination file."""
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        benchmark.save_json_result({"latency": float("nan")}, path)
    assert not path.exists()


def test_latency_measurement_reports_average_core_equivalents():
    """Measure process CPU time over the complete post-warmup sample block."""

    sample_clock = iter([10, 20])
    block_wall_clock = iter([100, 200])
    process_cpu_clock = iter([30, 90])

    measurement = benchmark.collect_latency_measurement(
        RecordingBackend([]),
        object(),
        benchmark.BenchmarkConfig(
            warmup=0,
            iterations=1,
        ),
        clock_ns=lambda: next(sample_clock),
        block_clock_ns=lambda: next(block_wall_clock),
        process_clock_ns=lambda: next(process_cpu_clock),
    )

    # The individual inference interval spans from 10 to 20 ns.
    assert measurement.samples_ns == (10,)

    # The outer sample block spans from 100 to 200 ns.
    assert measurement.sampling_block_wall_time_ns == 100

    # The process consumes 60 ns of aggregate CPU time during that block.
    assert measurement.sampling_block_process_cpu_time_ns == 60

    # Aggregate process CPU time divided by wall time gives an average
    # CPU-core equivalent, not an observed active-thread count.
    assert measurement.average_process_cpu_core_equivalents == pytest.approx(0.6)


@pytest.mark.parametrize(
    (
        "block_wall_ticks",
        "process_cpu_ticks",
        "expected_message",
    ),
    [
        ((100, 100), (20, 30), "wall time must be positive"),
        ((100, 200), (30, 20), "process CPU time cannot be negative"),
    ],
    ids=[
        "nonpositive_wall_time",
        "negative_process_cpu_time",
    ],
)
def test_latency_measurement_rejects_invalid_block_clocks(
    block_wall_ticks,
    process_cpu_ticks,
    expected_message,
):
    """Reject invalid outer timing intervals instead of persisting them."""

    sample_clock = iter([10, 20])
    block_wall_clock = iter(block_wall_ticks)
    process_cpu_clock = iter(process_cpu_ticks)

    with pytest.raises(ValueError, match=expected_message):
        benchmark.collect_latency_measurement(
            RecordingBackend([]),
            object(),
            benchmark.BenchmarkConfig(
                warmup=0,
                iterations=1,
            ),
            clock_ns=lambda: next(sample_clock),
            block_clock_ns=lambda: next(block_wall_clock),
            process_clock_ns=lambda: next(process_cpu_clock),
        )
