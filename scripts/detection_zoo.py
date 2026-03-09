
from __future__ import annotations

import argparse
import sys
import warnings
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
    parser = argparse.ArgumentParser(description="Detection local model zoo utilities (no downloads).")

    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring (case-insensitive).")
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument("--timeline", action="store_true", help="Print a best-effort detection family timeline (by year).")

    parser.add_argument("--smoke", type=str, default=None, metavar="ARCH_ID", help="Run a forward smoke on an arch id.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke inputs.")
    parser.add_argument("--image-size", type=int, default=64, help="Image size for smoke inputs.")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels for local detectors.")
    parser.add_argument("--num-classes", type=int, default=2, help="Detector classes.")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier for local detectors.")

    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()

    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\..*",
        category=FutureWarning,
    )

    from dlhub.vision.detection_zoo import build_local_model, list_local_arches

    args = parse_args()

    if not args.list and not args.timeline and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/detection_zoo.py --list")
        print("- python scripts/detection_zoo.py --timeline")
        print("- python scripts/detection_zoo.py --smoke dldet:ssd_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Detection local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.timeline:
        from dlhub.vision.detection._timeline import (
            ARCHIVE_END_YEAR,
            ARCHIVE_START_YEAR,
            entries,
            example_arch_id,
            family_series_label,
        )

        archive_entries = entries()
        if needle:
            archive_entries = [
                entry
                for entry in archive_entries
                if needle in entry.family.lower()
                or needle in entry.method.lower()
                or needle in entry.group.lower()
                or needle in family_series_label(entry.family).lower()
            ]

        timeline = sorted(
            archive_entries,
            key=lambda entry: (
                9999 if entry.year is None else int(entry.year),
                entry.group,
                entry.family,
            ),
        )

        print("Detection timeline (best-effort)")
        print(f"- total_families={len(timeline)}")
        print(f"- total_arches={len(arches)}")

        by_year: dict[int | str, list] = {}
        for entry in timeline:
            key: int | str = "unknown" if entry.year is None else int(entry.year)
            by_year.setdefault(key, []).append(entry)

        for year in range(ARCHIVE_START_YEAR, ARCHIVE_END_YEAR + 1):
            print("")
            print(year)
            year_entries = by_year.get(year, [])
            if not year_entries:
                print("- no archived family metadata yet")
                continue
            for entry in year_entries:
                series = family_series_label(entry.family)
                example = example_arch_id(entry.family)
                print(f"- {entry.family} [{entry.group}] {{{series}}}: {entry.method} -> {example}")

        if by_year.get("unknown"):
            print("")
            print("unknown")
            for entry in by_year["unknown"]:
                series = family_series_label(entry.family)
                example = example_arch_id(entry.family)
                print(f"- {entry.family} [{entry.group}] {{{series}}}: {entry.method} -> {example}")

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"dldet:{arch_id}"

        import torch

        x = torch.randn(int(args.batch_size), int(args.in_channels), int(args.image_size), int(args.image_size))
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_classes=int(args.num_classes),
            width_mult=float(args.width_mult),
        )
        model.eval()
        with torch.no_grad():
            out = model(x)

        print("")
        print(f"smoke: {arch_id}")
        print(f"- output={_summarize(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
