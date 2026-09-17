"""Shared loading and execution helpers for executable evaluation cases."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REQUIRED_FIELDS = {
    "id",
    "library",
    "source_version",
    "target_version",
    "prompt",
    "right_context",
    "assertions",
    "old_completion",
    "new_completion",
    "expected_old_error",
}


@dataclass
class ExecutionResult:
    """Result of running one rendered completion in a child process."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    completed_assertions: bool
    duration_seconds: float

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.returncode == 0 and self.completed_assertions

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _case_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        yield from sorted(path.glob("*.json"))
        return
    raise FileNotFoundError(f"Case path does not exist: {path}")


def load_cases(path: str | Path) -> List[Dict[str, Any]]:
    """Load and validate cases from one JSON file or a directory of JSON files."""

    cases: List[Dict[str, Any]] = []
    for case_file in _case_files(Path(path)):
        with case_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{case_file} must contain a JSON array")
        for index, case in enumerate(payload):
            if not isinstance(case, dict):
                raise ValueError(f"{case_file}[{index}] must be a JSON object")
            missing = sorted(REQUIRED_FIELDS - set(case))
            if missing:
                raise ValueError(
                    f"{case_file}[{index}] is missing fields: {', '.join(missing)}"
                )
            cases.append(case)

    ids = [case["id"] for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case ids: {', '.join(duplicates)}")
    return cases


def filter_by_library(
    cases: List[Dict[str, Any]], library: str
) -> List[Dict[str, Any]]:
    """Keep only cases belonging to one library (no-op when library is empty)."""

    if not library:
        return cases
    return [case for case in cases if case.get("library") == library]


def render_case(case: Mapping[str, Any], completion: str) -> str:
    """Insert a completion between the case's left and right contexts."""

    return "".join(
        [
            str(case["prompt"]),
            completion,
            str(case["right_context"]),
            "\n",
            str(case["assertions"]),
            "\n",
        ]
    )


def run_completion(
    case: Mapping[str, Any],
    completion: str,
    python_executable: str,
    timeout_seconds: float = 20.0,
) -> ExecutionResult:
    """Render and execute one completion in an isolated temporary directory."""

    source = render_case(case, completion)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="libevolutioneval-") as temp_dir:
        temp_path = Path(temp_dir)
        program_path = temp_path / "candidate.py"
        sentinel_path = temp_path / "assertions-completed"
        sentinel_source = (
            "\nfrom pathlib import Path as _LibEvolutionEvalPath\n"
            f"_LibEvolutionEvalPath({str(sentinel_path)!r}).write_text("
            "'ok', encoding='utf-8')\n"
        )
        program_path.write_text(source + sentinel_source, encoding="utf-8")
        mpl_config = temp_path / "matplotlib-config"
        mpl_config.mkdir()

        environment = os.environ.copy()
        environment.pop("PYTHONOPTIMIZE", None)
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(mpl_config),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        try:
            completed = subprocess.run(
                [python_executable, str(program_path)],
                cwd=temp_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(
                returncode=None,
                stdout=_as_text(error.stdout),
                stderr=_as_text(error.stderr),
                timed_out=True,
                completed_assertions=False,
                duration_seconds=time.monotonic() - start,
            )

        completed_assertions = (
            sentinel_path.exists()
            and sentinel_path.read_text(encoding="utf-8") == "ok"
        )

    return ExecutionResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
        completed_assertions=completed_assertions,
        duration_seconds=time.monotonic() - start,
    )


def write_json_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
