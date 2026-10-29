"""Generate auditable JSON and CSV summaries for the Day 6 thread sweep.

The command loads immutable per-process result files, delegates all campaign
validation and statistical aggregation to the analysis module, and writes new
summary artifacts without modifying or overwriting any raw experiment record.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from inference_bench.thread_sweep import (
    thread_sweep_csv_rows,
    validate_and_summarize_thread_sweep,
)


# Resolve default paths relative to the repository root.
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    ROOT
    / "artifacts/results/day6/formal/ac_v1/raw"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts/results/day6/formal/ac_v1"
)


def load_records(input_dir: Path) -> list[dict[str, object]]:
    """Load all raw JSON records from one result directory."""

    # Discover raw result files in deterministic filename order.
    paths = sorted(input_dir.glob("*.json"))

    records: list[dict[str, object]] = []

    for path in paths:
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)

        # Require every source file to contain a top-level JSON object.
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}: top-level JSON value must be an object"
            )

        records.append(record)

    return records


def write_summary(
    summary: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write strict JSON and CSV summaries without overwriting files."""

    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    # Refuse to replace any existing formal summary artifact.
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing Day 6 summary"
        )

    # Convert the hierarchical summary into a flat 15-row table.
    rows = thread_sweep_csv_rows(summary)

    if not rows:
        raise ValueError("Summary did not produce any CSV rows")

    # Serialize strict JSON and reject non-finite values before writing.
    json_content = json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the JSON output exclusively to preserve existing evidence.
    with json_path.open(
        "x",
        encoding="utf-8",
    ) as stream:
        stream.write(json_content + "\n")

    # Let the csv module manage platform-specific newline handling.
    with csv_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as stream:
        # Derive a stable column order from the first deterministic row.
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        # Write the header before emitting every configuration row.
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    """Parse source and destination directories for summary generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the 45 immutable raw JSON records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive summary.json and summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the formal campaign and write both summary formats."""

    args = parse_args()
    records = load_records(args.input_dir)
    summary = validate_and_summarize_thread_sweep(records)
    json_path, csv_path = write_summary(
        summary,
        args.output_dir,
    )

    print(
        f"Validated {summary['validation']['record_count']} "
        f"records across "
        f"{summary['validation']['group_count']} groups."
    )
    print(f"JSON summary: {json_path.relative_to(ROOT)}")
    print(f"CSV summary: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()