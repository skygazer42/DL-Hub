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
        description="Lane detection local model zoo utilities (no downloads)."
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
        "--in-channels", type=int, default=3, help="Input channels for local lane detectors."
    )
    parser.add_argument("--num-lanes", type=int, default=4, help="Number of lanes to predict.")
    parser.add_argument(
        "--width-mult", type=float, default=1.0, help="Width multiplier for local models."
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for local models.")
    parser.add_argument("--num-points", type=int, default=16, help="Points per lane curve.")
    parser.add_argument("--num-rows", type=int, default=16, help="Rows for row-anchor methods.")
    parser.add_argument("--grid-size", type=int, default=32, help="Grid size for row anchors.")
    parser.add_argument(
        "--num-anchors", type=int, default=24, help="Anchor count for anchor-based methods."
    )
    parser.add_argument(
        "--num-queries", type=int, default=6, help="Query count for transformer-style methods."
    )

    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\..*",
        category=FutureWarning,
    )

    from dlhub.vision.lane_detection_zoo import build_local_model, list_local_arches

    args = parse_args()

    if not args.list and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/lane_detection_zoo.py --list")
        print("- python scripts/lane_detection_zoo.py --search tiny --list")
        print("- python scripts/lane_detection_zoo.py --smoke dllane:lstr_tiny")
        return 2

    arches = list_local_arches()
    if args.search:
        needle = str(args.search).lower()
        arches = [arch for arch in arches if needle in arch.lower()]

    if args.list:
        print("Lane detection local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"dllane:{arch_id}"

        import torch

        x = torch.randn(
            int(args.batch_size), int(args.in_channels), int(args.image_size), int(args.image_size)
        )
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_lanes=int(args.num_lanes),
            image_size=int(args.image_size),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
            num_points=int(args.num_points),
            num_rows=int(args.num_rows),
            grid_size=int(args.grid_size),
            num_anchors=int(args.num_anchors),
            num_queries=int(args.num_queries),
        )
        model.eval()
        with torch.no_grad():
            out = model(x)

        print("")
        print(f"smoke: {arch_id}")
        print(f"- model={type(model).__name__}")
        print(f"- input_shape={tuple(x.shape)}")
        print(f"- output={_summarize(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
