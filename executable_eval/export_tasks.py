"""Export model-facing tasks without reference answers or hidden assertions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from executable_eval.core import filter_by_library, load_cases


PUBLIC_FIELDS = (
    "id",
    "library",
    "target_version",
    "api",
    "description",
    "prompt",
    "right_context",
)


def public_task(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {field: case[field] for field in PUBLIC_FIELDS if field in case}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export prompts while withholding reference completions and tests."
    )
    parser.add_argument("--cases", required=True, help="JSON case file or directory")
    parser.add_argument("--output", required=True, help="Destination JSONL path")
    parser.add_argument("--library", default="", help="Keep only cases of this library")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = filter_by_library(load_cases(args.cases), args.library)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(public_task(case), ensure_ascii=False) + "\n")
    print(f"Exported {len(cases)} tasks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

