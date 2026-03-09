
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
    parser = argparse.ArgumentParser(description="Action recognition local model zoo utilities (video + skeleton, no downloads).")
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring (case-insensitive).")
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument("--timeline", action="store_true", help="Print a best-effort action recognition timeline (by year).")

    parser.add_argument("--smoke", type=str, default=None, metavar="ARCH_ID", help="Run a forward smoke on an arch id.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke inputs.")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels (video=RGB=3, skeleton=xyz=3).")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of action classes.")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier for local models.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for local models.")

    # Video input params
    parser.add_argument("--image-size", type=int, default=64, help="Video spatial size for smoke inputs.")
    parser.add_argument("--frames", type=int, default=8, help="Video time dimension T for smoke inputs.")

    # Skeleton input params
    parser.add_argument("--num-joints", type=int, default=17, help="Skeleton joint count V for smoke inputs.")
    parser.add_argument("--seq-len", type=int, default=32, help="Skeleton sequence length T for smoke inputs.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()

    # Keep CLI output clean: importing torch may emit a noisy FutureWarning about `pynvml`.
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\..*",
        category=FutureWarning,
    )

    from dlhub.vision.action_recognition_zoo import build_local_model, list_local_arches

    args = parse_args()
    if not args.list and not args.timeline and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/action_recognition_zoo.py --list")
        print("- python scripts/action_recognition_zoo.py --timeline")
        print("- python scripts/action_recognition_zoo.py --smoke dlactv:c3d_tiny")
        print("- python scripts/action_recognition_zoo.py --smoke dlacts:stgcn_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        counts: dict[str, int] = {}
        for arch in arches:
            prefix = arch.split(":", 1)[0]
            counts[prefix] = counts.get(prefix, 0) + 1

        print("Action recognition local zoo")
        print(f"- total_arches={len(arches)}")
        for k in sorted(counts):
            print(f"- {k}={counts[k]}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.timeline:
        from dlhub.vision.action_recognition._timeline import by_family

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
                prefix = "dlactv" if entry.group == "video" else "dlacts"
                timeline.append((year, entry.group, entry.family, entry.method, f"{prefix}:{entry.family}_tiny"))

        timeline.sort(key=lambda x: (x[0], x[1], x[2]))

        print("Action recognition timeline (best-effort)")
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
        # Default to video prefix if user passes a bare variant like "c3d_tiny".
        if ":" not in arch_id:
            arch_id = f"dlactv:{arch_id}"

        prefix = arch_id.split(":", 1)[0].strip().lower()
        if prefix == "dlacts":
            x = torch.randn(int(args.batch_size), int(args.in_channels), int(args.seq_len), int(args.num_joints))
        else:
            x = torch.randn(int(args.batch_size), int(args.in_channels), int(args.frames), int(args.image_size), int(args.image_size))

        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_classes=int(args.num_classes),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
            image_size=int(args.image_size),
            frames=int(args.frames),
            num_joints=int(args.num_joints),
            seq_len=int(args.seq_len),
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

