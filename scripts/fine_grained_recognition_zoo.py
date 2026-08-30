import argparse
import sys
import warnings
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

    print_limited(lines, limit=limit, annotate_fidelity=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-grained visual recognition local model zoo utilities (no downloads)."
    )
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument(
        "--search", type=str, default=None, help="Filter list by substring (case-insensitive)."
    )
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Print a best-effort FGVC family timeline (by year).",
    )
    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        metavar="ARCH_ID",
        help="Run a forward smoke on an arch id.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke inputs.")
    parser.add_argument("--image-size", type=int, default=64, help="Image size for smoke inputs.")
    parser.add_argument(
        "--in-channels", type=int, default=3, help="Input channels for local models."
    )
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes.")
    parser.add_argument(
        "--width-mult", type=float, default=1.0, help="Width multiplier for local models."
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for local models.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()

    # Keep CLI output clean: importing torch may emit a noisy FutureWarning about `pynvml`.
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\..*",
        category=FutureWarning,
    )

    from dlhub.vision.fine_grained_recognition_zoo import build_local_model, list_local_arches

    args = parse_args()
    if not args.list and not args.timeline and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/fine_grained_recognition_zoo.py --list")
        print("- python scripts/fine_grained_recognition_zoo.py --timeline")
        print("- python scripts/fine_grained_recognition_zoo.py --smoke dlfgvc:transfg_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Fine-grained recognition local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.timeline:
        from dlhub.vision.fine_grained_recognition._timeline import by_family

        families: set[str] = set()
        for a in arches:
            name = a.split(":", 1)[1]
            for suf in ("_tiny", "_small", "_base"):
                if name.endswith(suf):
                    families.add(name[: -len(suf)])
                    break
            else:
                families.add(name)

        mapping = by_family()
        timeline = []
        for fam in sorted(families):
            if needle and needle not in fam.lower():
                continue
            entry = mapping.get(fam)
            if entry is None:
                timeline.append((9999, "unknown", fam, "unknown", "unknown"))
            else:
                year = 9999 if entry.year is None else int(entry.year)
                timeline.append(
                    (year, entry.group, entry.family, entry.method, f"dlfgvc:{entry.family}_tiny")
                )

        timeline.sort(key=lambda x: (x[0], x[1], x[2]))

        print("Fine-grained recognition timeline (best-effort)")
        print(f"- total_families={len(timeline)}")
        print(f"- total_arches={len(arches)}")

        current_year = None
        for year, group, family, method, example in timeline:
            y = "unknown" if year == 9999 else str(year)
            if y != current_year:
                print("")
                print(y)
                current_year = y
            print(f"- {family} [{group}]: {method} -> {example}")

    if args.smoke is not None:
        import torch

        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"dlfgvc:{arch_id}"
        x = torch.randn(
            int(args.batch_size), int(args.in_channels), int(args.image_size), int(args.image_size)
        )
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_classes=int(args.num_classes),
            image_size=int(args.image_size),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
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
