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
    rows = list(lines)
    limit = int(limit)
    if limit <= 0:
        return
    if len(rows) <= limit:
        for row in rows:
            print(row)
        return
    if limit <= 20:
        for row in rows[:limit]:
            print(row)
        print(f"... ({len(rows) - limit} more) ...")
        return
    head = limit - 10
    for row in rows[:head]:
        print(row)
    print(f"... ({len(rows) - limit} more) ...")
    for row in rows[-10:]:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face parsing local model zoo utilities (no downloads).")
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring.")
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument("--smoke", type=str, default=None, metavar="ARCH_ID", help="Run a forward smoke.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke inputs.")
    parser.add_argument("--image-size", type=int, default=64, help="Input image size for smoke inputs.")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels.")
    parser.add_argument("--num-classes", type=int, default=11, help="Number of face parsing classes.")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    from dlhub.vision.face_parsing_zoo import build_local_model, list_local_arches

    args = parse_args()
    if not args.list and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/face_parsing_zoo.py --list")
        print("- python scripts/face_parsing_zoo.py --search tiny --list")
        print("- python scripts/face_parsing_zoo.py --smoke fparse:roi_tanh_warp_tiny")
        return 2

    arches = list_local_arches()
    if args.search:
        needle = str(args.search).lower()
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Face parsing local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"fparse:{arch_id}"

        import torch

        image = torch.randn(
            int(args.batch_size),
            int(args.in_channels),
            int(args.image_size),
            int(args.image_size),
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
            out = model(image)

        print("")
        print(f"smoke: {arch_id}")
        print(f"- model={type(model).__name__}")
        print(f"- image_shape={tuple(image.shape)}")
        print(f"- output={_summarize(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
