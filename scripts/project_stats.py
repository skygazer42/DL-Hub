"""Compute repo statistics and sync the README/docs stat blocks.

Usage (from the repo root):
    python scripts/project_stats.py --json    # print stats as JSON
    python scripts/project_stats.py --check   # exit 1 if any stat block is stale
    python scripts/project_stats.py --write   # rewrite stale stat blocks in place
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    return repo_root


def main() -> int:
    repo_root = _ensure_repo_root_on_path()
    from dlhub import project_stats

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", action="store_true", help="print computed stats as JSON")
    group.add_argument("--check", action="store_true", help="fail if any stats block is stale")
    group.add_argument("--write", action="store_true", help="rewrite stale stats blocks")
    args = parser.parse_args()

    if args.json:
        stats = project_stats.compute_stats(repo_root)
        payload = dataclasses.asdict(stats)
        payload["lessons_total"] = stats.lessons_total
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.check:
        stale = project_stats.check_files(repo_root)
        if stale:
            print("Stale stats blocks in:", ", ".join(stale))
            print("Run: python scripts/project_stats.py --write")
            return 1
        print("All stats blocks are up to date.")
        return 0

    changed = project_stats.write_files(repo_root)
    if changed:
        print("Updated stats blocks in:", ", ".join(changed))
    else:
        print("All stats blocks were already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
