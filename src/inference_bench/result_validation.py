"""Shared validation for persisted experiment records."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from inference_bench.benchmark import summarize_latency


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def field(record: Mapping[str, Any], path: str) -> Any:
    """Return one required nested field using a dotted path."""

    value: Any = record

    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Missing required field: {path}")
        value = value[key]

    return value


def finite_number(value: Any, path: str) -> float:
    """Return one finite numeric value as a float."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path} must be a finite number")

    return float(value)


def required_string(value: Any, path: str) -> str:
    """Return one required nonempty string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")

    return value


def validate_correctness(record: Mapping[str, Any]) -> None:
    """Require correctness gates to pass before and after execution."""

    for phase in ("before", "after"):
        if field(record, f"correctness.{phase}.allclose") is not True:
            raise ValueError(
                f"Correctness allclose failed during {phase}"
            )
        if field(record, f"correctness.{phase}.top1_match") is not True:
            raise ValueError(
                f"Correctness Top-1 failed during {phase}"
            )


def validate_thread_environment(record: Mapping[str, Any]) -> None:
    """Require external thread-pool variables to remain unset."""

    for variable in THREAD_ENVIRONMENT_VARIABLES:
        if (
            field(
                record,
                f"environment.thread_environment.{variable}",
            )
            is not None
        ):
            raise ValueError("Formal thread environment must be unset")


def validate_ac_power(record: Mapping[str, Any]) -> None:
    """Require the recorded post-run power source to be AC power."""

    power_source = field(
        record,
        "environment.power_source_snapshot_after_run",
    )
    if not isinstance(power_source, str) or "AC Power" not in power_source:
        raise ValueError("Formal campaign requires AC Power")


def validate_statistics(
    record: Mapping[str, Any],
    samples: Sequence[int],
) -> dict[str, Any]:
    """Recompute latency statistics and verify their persisted values."""

    computed = asdict(
        summarize_latency(samples, batch_size=1)
    )
    stored = field(record, "statistics")

    if not isinstance(stored, Mapping):
        raise ValueError("statistics must be an object")
    if stored.get("count") != len(samples):
        raise ValueError("Stored sample count does not match raw samples")

    for name, computed_value in computed.items():
        if name == "count":
            continue

        stored_value = finite_number(
            stored.get(name),
            f"statistics.{name}",
        )
        if not math.isclose(
            stored_value,
            computed_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Stored {name} does not match raw samples"
            )

    return computed
