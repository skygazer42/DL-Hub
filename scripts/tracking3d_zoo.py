from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _summarize(obj) -> str:
    try:
        import torch
    except Exception:
        return f"{type(obj).__name__}"

    if isinstance(obj, torch.Tensor):
        return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
    if isinstance(obj, dict):
        keys = ", ".join(sorted(map(str, obj.keys())))
        return f"dict(keys=[{keys}])"
    if isinstance(obj, list | tuple):
        head = ", ".join(_summarize(x) for x in obj[:2])
        tail = "" if len(obj) <= 2 else f", ... (+{len(obj) - 2})"
        return f"{type(obj).__name__}([{head}{tail}])"
    return f"{type(obj).__name__}"


def _print_lines(lines: Iterable[str], *, limit: int = 80) -> None:
    lines = list(lines)
    if len(lines) <= limit:
        for line in lines:
            print(line)
        return
    head = max(10, limit - 10)
    for line in lines[:head]:
        print(line)
    print(f"... ({len(lines) - limit} more) ...")
    for line in lines[-10:]:
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracking3D local model zoo utilities (no downloads).")
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring.")
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument("--timeline", action="store_true", help="Print a best-effort Tracking3D timeline.")
    parser.add_argument("--smoke", type=str, default=None, metavar="ARCH_ID", help="Run a short sequence smoke on an arch id.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=4, help="Tracking sequence length.")
    parser.add_argument("--num-points", type=int, default=128, help="Number of points per frame.")
    parser.add_argument("--in-channels", type=int, default=3, help="Point feature channels.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of track classes.")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    from dlhub.pointcloud.tracking3d._timeline import entries
    from dlhub.pointcloud.tracking3d_zoo import build_local_model, list_local_arches

    args = parse_args()
    if not args.list and not args.timeline and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/tracking3d_zoo.py --list")
        print("- python scripts/tracking3d_zoo.py --timeline")
        print("- python scripts/tracking3d_zoo.py --smoke pctrk3d:ab3dmot_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Tracking3D local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.timeline:
        timeline = entries()
        if needle:
            timeline = [
                e for e in timeline if needle in e.family.lower() or needle in e.method.lower() or needle in e.group.lower()
            ]
        print("Tracking3D timeline")
        print(f"- total_families={len(timeline)}")
        print(f"- total_arches={len(arches)}")
        current_year = None
        for e in sorted(timeline, key=lambda x: (9999 if x.year is None else x.year, x.group, x.family)):
            y = "unknown" if e.year is None else str(e.year)
            if y != current_year:
                print("")
                print(y)
                current_year = y
            print(f"- {e.family} [{e.group}]: {e.method} -> pctrk3d:{e.family}_tiny")

    if args.smoke is not None:
        import torch

        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"pctrk3d:{arch_id}"
        x = torch.randn(int(args.batch_size), int(args.seq_len), int(args.num_points), int(args.in_channels))
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_classes=int(args.num_classes),
            seq_len=int(args.seq_len),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
        )
        out = model.track(x)
        print("")
        print(f"smoke: {arch_id}")
        print(f"- output={_summarize(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
