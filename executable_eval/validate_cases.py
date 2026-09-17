"""Validate that old completions fail and current completions pass."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Mapping

from executable_eval.core import (
    filter_by_library,
    load_cases,
    run_completion,
    write_json_report,
)


def validate_case(
    case: Mapping[str, Any],
    python_executable: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    old_result = run_completion(
        case, str(case["old_completion"]), python_executable, timeout_seconds
    )
    new_result = run_completion(
        case, str(case["new_completion"]), python_executable, timeout_seconds
    )
    expected_error = str(case["expected_old_error"])
    old_output = old_result.stdout + old_result.stderr
    old_failed_for_expected_reason = (
        not old_result.timed_out
        and old_result.returncode not in (None, 0)
        and expected_error in old_output
    )
    valid = old_failed_for_expected_reason and new_result.passed
    return {
        "id": case["id"],
        "valid": valid,
        "old_failed_for_expected_reason": old_failed_for_expected_reason,
        "expected_old_error": expected_error,
        "old": old_result.to_dict(),
        "new": new_result.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep only cases whose old API fails and new API passes."
    )
    parser.add_argument("--cases", required=True, help="JSON case file or directory")
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable for the target version"
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="Seconds per run")
    parser.add_argument("--json-report", help="Optional path for the full JSON report")
    parser.add_argument("--library", default="", help="Keep only cases of this library")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = filter_by_library(load_cases(args.cases), args.library)
    results = []
    for case in cases:
        result = validate_case(case, args.python, args.timeout)
        results.append(result)
        status = "PASS" if result["valid"] else "FAIL"
        print(
            f"{status} {case['id']} "
            f"old_expected_failure={result['old_failed_for_expected_reason']} "
            f"new_pass={result['new']['passed']}"
        )

    valid_count = sum(result["valid"] for result in results)
    summary = {
        "valid": valid_count,
        "total": len(results),
        "all_valid": valid_count == len(results),
        "results": results,
    }
    print(f"Validated {valid_count}/{len(results)} cases")
    if args.json_report:
        write_json_report(args.json_report, summary)
    return 0 if summary["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

