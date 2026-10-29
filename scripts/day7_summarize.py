"""Generate formal Day 7 graph-optimization and profiling summaries.

This command reads raw experiment records and writes the formal JSON and CSV
artifacts. Campaign validation and aggregation are implemented by
inference_bench.graph_optimization and inference_bench.profiling_campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from inference_bench.graph_optimization import (
    graph_optimization_csv_rows,
    validate_and_summarize_graph_optimization,
)
from inference_bench.profiling_campaign import (
    profiling_operator_csv_rows,
    validate_and_summarize_profiling_campaign,
    validate_cross_campaign,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "artifacts/results/day7/formal/ac_v1"
)
DEFAULT_GRAPH_INPUT_DIR = (
    DEFAULT_OUTPUT_ROOT
    / "graph_optimization/raw"
)
DEFAULT_PROFILE_INPUT_DIR = (
    DEFAULT_OUTPUT_ROOT
    / "profiling/raw"
)


def load_records(input_dir: Path) -> list[dict[str, object]]:
    """Load all JSON objects from one raw-result directory."""

    records: list[dict[str, object]] = []

    # Filename order is deterministic even though campaign validators index
    # records by their explicit configuration fields (level/threads).
    for path in sorted(input_dir.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)

        if not isinstance(record, dict):
            raise ValueError(
                f"{path}: top-level JSON value must be an object"
            )

        records.append(record)

    return records


def write_json_exclusive(
    value: Mapping[str, Any],
    path: Path,
) -> None:
    """Write strict formatted JSON without overwriting an artifact."""

    # Reject non-finite values because the formal output must remain portable
    # standards-compliant JSON.
    content = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    # Exclusive creation preserves any previously published summary artifact.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content + "\n")


def write_csv_exclusive(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    path: Path,
) -> None:
    """Write one CSV file without overwriting an artifact."""

    if not rows:
        raise ValueError(f"No rows generated for {path.name}")

    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(
    graph_optimization_summary: dict[str, Any],
    operator_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    output_root: Path,
) -> list[Path]:
    """Write the five formal summary artifacts exclusively."""

    graph_output_dir = output_root / "graph_optimization"
    profile_output_dir = output_root / "profiling"

    # Preserve the validated 24-process benchmark campaign, its eight
    # configuration summaries, and the derived graph and thread comparisons.
    graph_json_path = graph_output_dir / "summary.json"

    # Flatten the eight graph-level and thread-count configurations for tabular
    # review while retaining each independent round mean.
    graph_csv_path = graph_output_dir / "summary.csv"

    # Preserve the validated ten-profile campaign, per-profile totals, normalized
    # operator rows, and cross-campaign identity evidence.
    operator_json_path = profile_output_dir / "operator_summary.json"

    # Expose the stable operator columns used to compare PyTorch and ORT profiles.
    operator_csv_path = profile_output_dir / "operator_summary.csv"

    # Compare the original ONNX graph with the four optimized graphs and preserve
    # their operator, domain, node, initializer, and file-size changes.
    optimized_graph_json_path = profile_output_dir / "graph_summary.json"

    paths = [
        graph_json_path,
        graph_csv_path,
        operator_json_path,
        operator_csv_path,
        optimized_graph_json_path,
    ]

    existing = [path for path in paths if path.exists()]
    # Check the complete output set before creating any file so an existing
    # artifact cannot cause a partially refreshed summary set.
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing Day 7 summaries: "
            + ", ".join(str(path) for path in existing)
        )

    graph_rows = graph_optimization_csv_rows(
        graph_optimization_summary
    )
    operator_rows = profiling_operator_csv_rows(
        operator_summary
    )

    # Preflight all JSON payloads before creating output directories or files;
    # serialization errors therefore cannot leave partial formal artifacts.
    json.dumps(
        graph_optimization_summary,
        allow_nan=False,
    )
    json.dumps(
        operator_summary,
        allow_nan=False,
    )
    json.dumps(
        graph_summary,
        allow_nan=False,
    )

    graph_output_dir.mkdir(parents=True, exist_ok=True)
    profile_output_dir.mkdir(parents=True, exist_ok=True)

    write_json_exclusive(
        graph_optimization_summary,
        graph_json_path,
    )
    write_csv_exclusive(
        graph_rows,
        list(graph_rows[0]),
        graph_csv_path,
    )
    write_json_exclusive(
        operator_summary,
        operator_json_path,
    )
    write_csv_exclusive(
        operator_rows,
        list(operator_rows[0]),
        operator_csv_path,
    )
    write_json_exclusive(
        graph_summary,
        optimized_graph_json_path,
    )

    return paths


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""

    resolved = path.resolve()

    if resolved.is_relative_to(ROOT):
        return resolved.relative_to(ROOT).as_posix()

    return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    """Parse raw input and formal output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-input-dir",
        type=Path,
        default=DEFAULT_GRAPH_INPUT_DIR,
        help="Directory containing the 24 formal graph JSON records.",
    )
    parser.add_argument(
        "--profile-input-dir",
        type=Path,
        default=DEFAULT_PROFILE_INPUT_DIR,
        help="Directory containing the 10 profiling JSON records.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Day 7 formal ac_v1 output directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate both campaigns and write the five summaries."""

    args = parse_args()

    graph_records = load_records(args.graph_input_dir)
    profile_records = load_records(args.profile_input_dir)

    graph_optimization_summary = (
        validate_and_summarize_graph_optimization(
            graph_records
        )
    )
    (
        operator_summary,
        graph_summary,
        profile_fingerprint,
    ) = validate_and_summarize_profiling_campaign(
        profile_records
    )

    cross_campaign = validate_cross_campaign(
        graph_optimization_summary,
        profile_fingerprint,
        graph_summary,
    )

    # Embed the shared identity check in both profiling outputs so each
    # artifact remains independently reviewable.
    operator_summary["cross_campaign_validation"] = cross_campaign
    graph_summary["cross_campaign_validation"] = cross_campaign

    output_paths = write_summaries(
        graph_optimization_summary,
        operator_summary,
        graph_summary,
        args.output_root,
    )

    print(
        "Validated "
        f"{graph_optimization_summary['validation']['record_count']} "
        "formal latency records and "
        f"{operator_summary['validation']['record_count']} "
        "profiling records."
    )

    for path in output_paths:
        print(f"Summary written to: {display_path(path)}")


if __name__ == "__main__":
    main()
