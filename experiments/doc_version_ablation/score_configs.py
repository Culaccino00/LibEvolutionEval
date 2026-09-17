#!/usr/bin/env python3
"""Score the four doc-version ablation configs per library."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from executable_eval.core import load_cases  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = OUT_DIR / "predictions"
RESULTS_DIR = OUT_DIR / "results"

VENVS = {
    "matplotlib": ROOT / ".venv-mpl383/bin/python",
    "torch": ROOT / ".venv-torch220/bin/python",
}
CONFIGS = ("correct", "wrong", "none", "mixed")


def main() -> int:
    cases = load_cases(ROOT / "executable_eval/cases")
    libraries = sorted({case["library"] for case in cases})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, str]] = {}
    for config in CONFIGS:
        predictions_path = PREDICTIONS_DIR / f"{config}.jsonl"
        predictions = {}
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    predictions[item["id"]] = item["completion"]

        summary[config] = {}
        for library in libraries:
            library_ids = {case["id"] for case in cases if case["library"] == library}
            split_path = RESULTS_DIR / f"{config}_{library}_predictions.jsonl"
            with split_path.open("w", encoding="utf-8") as handle:
                for case_id in sorted(library_ids):
                    handle.write(
                        json.dumps(
                            {"id": case_id, "completion": predictions.get(case_id, "")},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            report_path = RESULTS_DIR / f"{config}_{library}_score.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "executable_eval.score_predictions",
                    "--cases",
                    str(ROOT / "executable_eval/cases"),
                    "--library",
                    library,
                    "--predictions",
                    str(split_path),
                    "--python",
                    str(VENVS[library]),
                    "--json-report",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            match = re.search(r"Score: (\d+)/(\d+)", completed.stdout)
            score = match.group(0) if match else f"ERROR: {completed.stdout[-200:]}{completed.stderr[-200:]}"
            summary[config][library] = score
            print(f"{config:8s} {library:12s} {score}", flush=True)

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nSummary written to {RESULTS_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
