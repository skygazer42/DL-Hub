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

    print_limited(lines, limit=limit, annotate_fidelity=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Panoptic segmentation local model zoo utilities (no downloads)."
    )

    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument(
        "--search", type=str, default=None, help="Filter list by substring (case-insensitive)."
    )
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")

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
    parser.add_argument("--num-thing-classes", type=int, default=3, help="Number of thing classes.")
    parser.add_argument("--num-stuff-classes", type=int, default=2, help="Number of stuff classes.")
    parser.add_argument(
        "--width-mult", type=float, default=1.0, help="Width multiplier for local models."
    )

    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()

    from dlhub.vision.panoptic_segmentation_zoo import build_local_model, list_local_arches

    args = parse_args()

    if not args.list and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/panoptic_segmentation_zoo.py --list")
        print("- python scripts/panoptic_segmentation_zoo.py --smoke dlpan:panoptic_fpn_tiny")
        return 2

    arches = list_local_arches()
    if args.search:
        needle = str(args.search).lower()
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Panoptic segmentation local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"dlpan:{arch_id}"

        import torch

        x = torch.randn(
            int(args.batch_size), int(args.in_channels), int(args.image_size), int(args.image_size)
        )
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_thing_classes=int(args.num_thing_classes),
            num_stuff_classes=int(args.num_stuff_classes),
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
