"""Score generated completions by executing their assertions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from executable_eval.core import (
    filter_by_library,
    load_cases,
    run_completion,
    write_json_report,
)


def load_predictions(path: str | Path) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or "id" not in item or "completion" not in item:
                raise ValueError(
                    f"Prediction line {line_number} must contain id and completion"
                )
            case_id = str(item["id"])
            if case_id in predictions:
                raise ValueError(f"Duplicate prediction id: {case_id}")
            predictions[case_id] = str(item["completion"])
    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute model completions in the target library environment."
    )
    parser.add_argument("--cases", required=True, help="JSON case file or directory")
    parser.add_argument("--predictions", required=True, help="JSONL model predictions")
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable for the target version"
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="Seconds per run")
    parser.add_argument("--json-report", help="Optional path for the full JSON report")
    parser.add_argument("--library", default="", help="Keep only cases of this library")
    return parser


def aggregate_by_change_type(
    cases: list[Dict[str, Any]], results: list[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    case_by_id = {str(case["id"]): case for case in cases}
    grouped: Dict[str, list[bool]] = defaultdict(list)
    for result in results:
        change_type = str(case_by_id[result["id"]].get("change_type", "unspecified"))
        grouped[change_type].append(bool(result["passed"]))
    return {
        change_type: {
            "passed": sum(outcomes),
            "total": len(outcomes),
            "pass_rate": sum(outcomes) / len(outcomes),
        }
        for change_type, outcomes in sorted(grouped.items())
    }


def main() -> int:
    args = build_parser().parse_args()
    cases = filter_by_library(load_cases(args.cases), args.library)
    predictions = load_predictions(args.predictions)
    known_ids = {str(case["id"]) for case in cases}
    unknown_ids = sorted(set(predictions) - known_ids)
    if unknown_ids:
        raise ValueError(f"Unknown prediction ids: {', '.join(unknown_ids)}")

    results: list[Dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        if case_id not in predictions:
            result: Dict[str, Any] = {
                "id": case_id,
                "passed": False,
                "missing": True,
            }
        else:
            execution = run_completion(
                case, predictions[case_id], args.python, args.timeout
            )
            result = {
                "id": case_id,
                "passed": execution.passed,
                "missing": False,
                "execution": execution.to_dict(),
            }
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case_id}")

    passed = sum(result["passed"] for result in results)
    total = len(results)
    summary = {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "by_change_type": aggregate_by_change_type(cases, results),
        "results": results,
    }
    print(f"Score: {passed}/{total} ({summary['pass_rate']:.1%})")
    for change_type, group in summary["by_change_type"].items():
        print(
            f"  {change_type}: {group['passed']}/{group['total']} "
            f"({group['pass_rate']:.1%})"
        )
    if args.json_report:
        write_json_report(args.json_report, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
