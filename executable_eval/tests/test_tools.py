import json
import sys
import tempfile
import unittest
from pathlib import Path

from executable_eval.core import load_cases, render_case, run_completion
from executable_eval.export_tasks import public_task
from executable_eval.score_predictions import aggregate_by_change_type, load_predictions
from executable_eval.validate_cases import validate_case


class ExecutableEvalTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "synthetic_version_change",
            "library": "stdlib",
            "source_version": "old",
            "target_version": "current",
            "prompt": "value = ",
            "right_context": "\n",
            "assertions": "assert value == 42",
            "old_completion": "1 / 0",
            "new_completion": "40 + 2",
            "expected_old_error": "division by zero",
        }

    def test_render_case_places_completion_between_contexts(self):
        source = render_case(self.case, "40 + 2")
        self.assertIn("value = 40 + 2\n", source)
        self.assertTrue(source.endswith("assert value == 42\n"))

    def test_runner_executes_assertions(self):
        result = run_completion(self.case, "40 + 2", sys.executable)
        self.assertTrue(result.passed, result.stderr)

    def test_clean_early_exit_does_not_bypass_assertions(self):
        result = run_completion(
            self.case, "__import__('sys').exit(0)", sys.executable
        )
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.completed_assertions)
        self.assertFalse(result.passed)

    def test_validator_requires_expected_old_error_and_new_success(self):
        result = validate_case(self.case, sys.executable, 5.0)
        self.assertTrue(result["valid"])
        self.assertTrue(result["old_failed_for_expected_reason"])
        self.assertTrue(result["new"]["passed"])

    def test_loaders_reject_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir) / "cases.json"
            case_path.write_text(json.dumps([self.case, self.case]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate case ids"):
                load_cases(case_path)

            predictions_path = Path(temp_dir) / "predictions.jsonl"
            prediction = {"id": self.case["id"], "completion": "40 + 2"}
            predictions_path.write_text(
                json.dumps(prediction) + "\n" + json.dumps(prediction) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate prediction id"):
                load_predictions(predictions_path)

    def test_public_task_does_not_leak_answers_or_assertions(self):
        task = public_task(self.case)
        self.assertEqual(task["id"], self.case["id"])
        self.assertNotIn("old_completion", task)
        self.assertNotIn("new_completion", task)
        self.assertNotIn("assertions", task)

    def test_reference_predictions_cover_all_pilot_cases(self):
        eval_dir = Path(__file__).resolve().parents[1]
        cases = load_cases(eval_dir / "cases")
        libraries = sorted({case["library"] for case in cases})
        self.assertTrue(libraries)
        for library in libraries:
            library_ids = {
                case["id"] for case in cases if case["library"] == library
            }
            for tag in ("new", "old"):
                filename = f"reference_{tag}_predictions_{library}.jsonl"
                predictions = load_predictions(eval_dir / "examples" / filename)
                self.assertEqual(set(predictions), library_ids)

    def test_rejected_candidates_are_not_accepted_cases(self):
        eval_dir = Path(__file__).resolve().parents[1]
        cases = load_cases(eval_dir / "cases")
        accepted_apis = {case.get("api") for case in cases}
        rejected = json.loads(
            (eval_dir / "candidate_review.json").read_text(encoding="utf-8")
        )
        self.assertTrue(rejected)
        self.assertTrue(all(item["decision"] == "rejected" for item in rejected))
        self.assertTrue(all(item["api"] not in accepted_apis for item in rejected))

    def test_change_type_aggregation(self):
        cases = [
            {"id": "a", "change_type": "removed_api"},
            {"id": "b", "change_type": "removed_api"},
            {"id": "c", "change_type": "renamed_parameter"},
        ]
        results = [
            {"id": "a", "passed": True},
            {"id": "b", "passed": False},
            {"id": "c", "passed": True},
        ]
        summary = aggregate_by_change_type(cases, results)
        self.assertEqual(summary["removed_api"]["passed"], 1)
        self.assertEqual(summary["removed_api"]["total"], 2)
        self.assertEqual(summary["renamed_parameter"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
