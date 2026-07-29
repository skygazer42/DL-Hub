from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _summarize(obj) -> str:
    from dlhub.cli_utils import summarize_output

    return summarize_output(obj)


def _print_lines(lines: Iterable[str], *, limit: int = 80) -> None:
    from dlhub.cli_utils import print_limited

    print_limited(lines, limit=limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated learning local zoo utilities (no networking)."
    )
    parser.add_argument("--list", action="store_true", help="List available strategy ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring.")
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument(
        "--timeline", action="store_true", help="Print a best-effort federated timeline."
    )
    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        metavar="ARCH_ID",
        help="Run a simulated round for an arch id.",
    )
    parser.add_argument("--param-dim", type=int, default=16, help="Parameter vector dimension.")
    parser.add_argument(
        "--num-clients", type=int, default=4, help="Number of participating clients."
    )
    parser.add_argument(
        "--local-steps", type=int, default=2, help="Nominal number of client local steps."
    )
    parser.add_argument(
        "--width-mult", type=float, default=1.0, help="Width multiplier for strategy internals."
    )
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    from dlhub.federated._timeline import entries
    from dlhub.federated_zoo import build_local_strategy, list_local_arches

    args = parse_args()
    if not args.list and not args.timeline and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/federated_zoo.py --list")
        print("- python scripts/federated_zoo.py --timeline")
        print("- python scripts/federated_zoo.py --smoke dlfed:fedavg_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Federated learning local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.timeline:
        timeline = entries()
        if needle:
            timeline = [
                entry
                for entry in timeline
                if needle in entry.family.lower()
                or needle in entry.method.lower()
                or needle in entry.group.lower()
            ]

        print("Federated learning timeline")
        print(f"- total_families={len(timeline)}")
        print(f"- total_arches={len(arches)}")
        current_year = None
        for entry in sorted(
            timeline, key=lambda x: (9999 if x.year is None else x.year, x.group, x.family)
        ):
            y = "unknown" if entry.year is None else str(entry.year)
            if y != current_year:
                print("")
                print(y)
                current_year = y
            print(f"- {entry.family} [{entry.group}]: {entry.method} -> dlfed:{entry.family}_tiny")

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"dlfed:{arch_id}"
        strategy = build_local_strategy(
            arch_id,
            param_dim=int(args.param_dim),
            num_clients=int(args.num_clients),
            local_steps=int(args.local_steps),
            width_mult=float(args.width_mult),
        )
        out = strategy.simulate_round(seed=0)
        print("")
        print(f"smoke: {arch_id}")
        print(f"- output={_summarize(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
