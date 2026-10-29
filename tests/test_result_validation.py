"""Unit tests for shared experiment-record validation."""

from __future__ import annotations

from dataclasses import asdict

from inference_bench.benchmark import summarize_latency
from inference_bench.result_validation import (
    field,
    validate_statistics,
)


def test_validate_statistics_recomputes_raw_latency_samples():
    samples = [1_000_000, 2_000_000, 3_000_000]
    record = {
        "statistics": asdict(
            summarize_latency(samples, batch_size=1)
        )
    }

    computed = validate_statistics(record, samples)

    assert field(record, "statistics.count") == 3
    assert computed["mean_ms"] == 2.0
