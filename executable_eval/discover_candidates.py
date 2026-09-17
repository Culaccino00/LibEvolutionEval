"""List API changes in the packaged LibEvolutionEval documentation snapshots."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any, Dict


def load_snapshot(
    archive_path: str | Path, library: str, version: str
) -> Dict[str, Dict[str, Any]]:
    member_name = f"{library}/{version.replace('.', '_')}.jsonl"
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"Missing archive member: {member_name}")
        entries: Dict[str, Dict[str, Any]] = {}
        for raw_line in member:
            item = json.loads(raw_line)
            entries[str(item["API_Call"])] = item
    return entries


def discover_changes(
    old: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]
) -> list[Dict[str, str]]:
    changes: list[Dict[str, str]] = []
    for api_name in sorted(set(old) | set(new)):
        old_item = old.get(api_name)
        new_item = new.get(api_name)
        if old_item is None:
            kind = "introduced"
        elif new_item is None:
            kind = "removed"
        elif old_item.get("Signature") != new_item.get("Signature"):
            kind = "signature_changed"
        else:
            continue
        changes.append(
            {
                "kind": kind,
                "api": api_name,
                "old_signature": "" if old_item is None else old_item.get("Signature", ""),
                "new_signature": "" if new_item is None else new_item.get("Signature", ""),
            }
        )
    return changes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find version-change candidates; execution tests are still required."
    )
    parser.add_argument(
        "--archive", default="data/package_apis.tar.gz", help="Documentation archive"
    )
    parser.add_argument("--library", default="matplotlib")
    parser.add_argument("--old-version", required=True)
    parser.add_argument("--new-version", required=True)
    parser.add_argument(
        "--kind",
        choices=("all", "introduced", "removed", "signature_changed"),
        default="all",
    )
    parser.add_argument("--contains", default="", help="Optional API-name substring")
    parser.add_argument("--limit", type=int, default=0, help="Zero means no limit")
    parser.add_argument("--jsonl", action="store_true", help="Emit JSONL instead of TSV")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    old = load_snapshot(args.archive, args.library, args.old_version)
    new = load_snapshot(args.archive, args.library, args.new_version)
    changes = discover_changes(old, new)
    if args.kind != "all":
        changes = [change for change in changes if change["kind"] == args.kind]
    if args.contains:
        changes = [change for change in changes if args.contains in change["api"]]
    if args.limit:
        changes = changes[: args.limit]

    for change in changes:
        if args.jsonl:
            print(json.dumps(change, ensure_ascii=False))
        else:
            print(
                "\t".join(
                    [
                        change["kind"],
                        change["api"],
                        change["old_signature"],
                        change["new_signature"],
                    ]
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

