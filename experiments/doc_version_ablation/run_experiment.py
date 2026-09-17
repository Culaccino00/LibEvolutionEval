#!/usr/bin/env python3
"""Doc-version ablation over the executable_eval benchmark.

Four configurations per task:
  correct : documentation from the target version (matplotlib 3.8.3 / torch 2.2.0)
  wrong   : documentation from the case's source version (old API still worked)
  none    : no documentation
  mixed   : both, each labeled with its version, order seeded-shuffled

The model (deepseek-v4-flash via the OpenAI-compatible DeepSeek endpoint)
writes the missing completion; predictions are scored with
executable_eval.score_predictions under the matching pinned venv.

Usage:
  python3 experiments/doc_version_ablation/run_experiment.py --dry-run
  python3 experiments/doc_version_ablation/run_experiment.py --run
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from executable_eval.core import load_cases  # noqa: E402
from executable_eval.discover_candidates import load_snapshot  # noqa: E402

ARCHIVE = ROOT / "data/package_apis.tar.gz"
OUT_DIR = Path(__file__).resolve().parent
RESPONSES_DIR = OUT_DIR / "responses"
PREDICTIONS_DIR = OUT_DIR / "predictions"

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_TOKENS = 8192  # endpoint cap for this model; 256k output is not supported
CONFIGS = ("correct", "wrong", "none", "mixed")

SYSTEM_MESSAGE = (
    "You complete Python code. The user shows a function with `<INSERT>` marking "
    "a missing fragment. Reply with only the code that should replace `<INSERT>`. "
    "No explanation, no markdown fences, no surrounding code."
)


def read_api_key() -> str:
    text = (ROOT / ".env").read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() in ("DEEPSEEK_API_KEY", "API_KEY", "TOKEN"):
                return value.strip().strip('"').strip("'")
            return value.strip().strip('"').strip("'")
        return line
    raise ValueError("No API key found in .env")


def clean_signature(sig: str) -> str:
    if not isinstance(sig, str):
        return ""
    return (
        sig.replace("[source]\u00b6", "")
        .replace("[source]", "")
        .replace("\u00b6", "")
        .replace("\u2192", "->")
        .strip()
    )


def format_doc(api_call: str, item: dict, version_label: str) -> str:
    signature = clean_signature(item.get("Signature", ""))
    description = str(item.get("Detailed_Description", "") or "")[:600]
    params = item.get("Parameters") or {}
    inputs = ", ".join(params) if isinstance(params, dict) and params else ""
    lines = [
        f"--- Documentation for `{api_call}` ({version_label}) ---",
        f"{api_call} : {signature}",
    ]
    if inputs:
        lines.append(f"Inputs: {inputs}")
    if description and description != "None":
        lines.append(description)
    return "\n".join(lines)


def find_doc(api: str, snapshot: dict) -> tuple[str, dict] | None:
    if api in snapshot:
        return api, snapshot[api]
    # progressively match parent path (method -> class -> module attr)
    parts = api.split(".")
    for cut in range(1, len(parts)):
        candidate = ".".join(parts[:-cut])
        if candidate in snapshot:
            return candidate, snapshot[candidate]
    # suffix match for completions written with short names
    matches = [key for key in snapshot if key.endswith("." + api)]
    if matches:
        best = min(matches, key=len)
        return best, snapshot[best]
    return None


CALLEE_RE = re.compile(r"([A-Za-z_][\w\.]*)\s*\(")


def primary_callee(completion: str) -> str | None:
    match = CALLEE_RE.search(completion)
    return match.group(1) if match else None


AVAILABLE_VERSIONS = {
    "matplotlib": ["3.0.3", "3.2.0", "3.3.4", "3.5.2", "3.6.3", "3.8.3"],
    "torch": [
        "0.4.0", "1.1.0", "1.2.0", "1.4.0", "1.6.0", "1.8.0",
        "1.10.0", "1.12.0", "1.13.0", "2.0.0", "2.2.0",
    ],
}


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def nearest_snapshot_version(library: str, version: str) -> str:
    """Newest packaged snapshot not newer than `version` (else the oldest)."""

    available = AVAILABLE_VERSIONS[library]
    older = [v for v in available if version_key(v) <= version_key(version)]
    return max(older, key=version_key) if older else min(available, key=version_key)


def snapshot_name(library: str, version: str) -> str:
    return f"v_{version}" if library == "torch" else version


class DocProvider:
    def __init__(self, cases: list[dict]) -> None:
        self.snapshots: dict[tuple[str, str], dict] = {}
        needed = set()
        for case in cases:
            needed.add((case["library"], case["target_version"]))
            needed.add(
                (case["library"], nearest_snapshot_version(case["library"], case["source_version"]))
            )
        for library, version in sorted(needed):
            self.snapshots[(library, version)] = load_snapshot(
                ARCHIVE, library, snapshot_name(library, version)
            )

    def doc(self, library: str, version: str, api: str) -> str | None:
        version = nearest_snapshot_version(library, version)
        found = find_doc(api, self.snapshots[(library, version)])
        if found is None:
            return None
        api_call, item = found
        return format_doc(api_call, item, f"{library} {version}")


def build_docs(config: str, case: dict, docs: DocProvider, rng: random.Random) -> str:
    library = case["library"]
    target = case["target_version"]
    source = case["source_version"]
    new_api = primary_callee(case["new_completion"]) or case["api"]
    old_api = primary_callee(case["old_completion"]) or case["api"]

    correct_doc = docs.doc(library, target, new_api) or docs.doc(library, target, case["api"])
    wrong_doc = docs.doc(library, source, old_api) or docs.doc(library, source, case["api"])

    if config == "none":
        return ""
    if config == "correct":
        return (correct_doc or "(no matching documentation found)") + "\n\n"
    if config == "wrong":
        return (wrong_doc or "(no matching documentation found)") + "\n\n"
    # mixed: both, seeded order
    pair = [doc for doc in (wrong_doc, correct_doc) if doc]
    rng.shuffle(pair)
    return "\n\n".join(pair) + "\n\n"


def build_user_message(config: str, case: dict, docs: DocProvider, rng: random.Random) -> str:
    doc_block = build_docs(config, case, docs, rng)
    code = f"{case['prompt']}<INSERT>{case['right_context']}"
    return (
        f"{doc_block}"
        "Complete the following Python code by replacing `<INSERT>`:\n\n"
        f"```python\n{code}\n```\n\n"
        "Reply with only the replacement code for `<INSERT>`."
    )


FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_completion(text: str) -> str:
    fence = FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    return text.strip("\n")


def call_api(session_key: str, user_message: str, max_retries: int = 4) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {session_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            if error.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    raise RuntimeError("unreachable")


def run_config(config: str, cases: list[dict], docs: DocProvider, api_key: str, workers: int) -> None:
    config_dir = RESPONSES_DIR / config
    config_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260916)

    messages = {
        case["id"]: build_user_message(config, case, docs, rng) for case in cases
    }

    def one(case_id: str) -> tuple[str, str]:
        cache = config_dir / f"{case_id}.json"
        if cache.exists():
            record = json.loads(cache.read_text(encoding="utf-8"))
            return case_id, record["completion"]
        raw = call_api(api_key, messages[case_id])
        completion = extract_completion(raw)
        cache.write_text(
            json.dumps({"id": case_id, "raw": raw, "completion": completion}, ensure_ascii=False),
            encoding="utf-8",
        )
        return case_id, completion

    completions: dict[str, str] = {}
    todo = [case["id"] for case in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, cid): cid for cid in todo}
        done = 0
        for future in as_completed(futures):
            cid = futures[future]
            try:
                case_id, completion = future.result()
                completions[case_id] = completion
            except Exception as error:  # noqa: BLE001
                print(f"ERROR {config}/{cid}: {error}", flush=True)
                completions[cid] = ""
            done += 1
            if done % 25 == 0:
                print(f"  {config}: {done}/{len(todo)}", flush=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{config}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {"id": case["id"], "completion": completions.get(case["id"], "")},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {path} ({sum(1 for c in completions.values() if c)} non-empty)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build prompts, no API calls")
    parser.add_argument("--run", action="store_true", help="Call the API")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args()

    cases = load_cases(ROOT / "executable_eval/cases")
    docs = DocProvider(cases)
    print(f"Loaded {len(cases)} cases and {len(docs.snapshots)} snapshots")

    configs = [c for c in args.configs.split(",") if c]
    if args.dry_run:
        rng = random.Random(20260916)
        for config in configs:
            hits = misses = 0
            total_chars = 0
            for case in cases:
                doc_block = build_docs(config, case, docs, rng)
                if "(no matching documentation found)" in doc_block:
                    misses += 1
                else:
                    hits += 1
                total_chars += len(doc_block)
            print(
                f"  {config}: doc hits {hits}, misses {misses}, "
                f"avg doc chars {total_chars // len(cases)}"
            )
        sample = next(c for c in cases if c["library"] == "torch")
        print("\n--- sample mixed prompt (first torch case) ---")
        print(build_user_message("mixed", sample, docs, random.Random(1))[:1500])
        return 0

    if args.run:
        api_key = read_api_key()
        for config in configs:
            print(f"=== config {config} ===", flush=True)
            run_config(config, cases, docs, api_key, args.workers)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
