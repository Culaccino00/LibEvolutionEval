"""Regenerate the old and current reference-prediction JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from executable_eval.core import filter_by_library, load_cases


def write_predictions(
    cases: list[Mapping[str, Any]], field: str, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            record = {"id": case["id"], "completion": case[field]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reference prediction files from the reviewed cases."
    )
    parser.add_argument("--cases", required=True, help="JSON case file or directory")
    parser.add_argument("--output-dir", required=True, help="Destination directory")
    parser.add_argument(
        "--library",
        default="",
        help="Keep only cases of this library and suffix output names with it",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = load_cases(args.cases)
    suffix = ""
    if args.library:
        cases = filter_by_library(cases, args.library)
        suffix = f"_{args.library}"
    output_dir = Path(args.output_dir)
    write_predictions(
        cases,
        "new_completion",
        output_dir / f"reference_new_predictions{suffix}.jsonl",
    )
    write_predictions(
        cases,
        "old_completion",
        output_dir / f"reference_old_predictions{suffix}.jsonl",
    )
    print(f"Generated old and current references for {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

