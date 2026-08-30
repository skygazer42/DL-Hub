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
        description="Style transfer local model zoo utilities (no downloads)."
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
        "--in-channels", type=int, default=3, help="Input channels for local stylizers."
    )
    parser.add_argument(
        "--width-mult", type=float, default=1.0, help="Width multiplier for local models."
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for local models.")

    # Optional builder kwargs. These are ignored by builders that don't accept them.
    parser.add_argument(
        "--steps", type=int, default=6, help="Sampling/optimization steps (method-dependent)."
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="Noise strength for diffusion img2img stylizers.",
    )
    parser.add_argument(
        "--style-dim", type=int, default=64, help="Style embedding dimension (method-dependent)."
    )
    parser.add_argument(
        "--control-scale", type=float, default=1.0, help="ControlNet hint scale (method-dependent)."
    )
    parser.add_argument("--num-layers", type=int, default=2, help="Layer count (method-dependent).")
    parser.add_argument(
        "--guidance-scale", type=float, default=2.0, help="CFG scale for diffusion stylizers."
    )
    parser.add_argument(
        "--ref-weight",
        type=float,
        default=1.0,
        help="Reference attention weight (method-dependent).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Attention temperature (method-dependent)."
    )
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    from dlhub.vision.style_transfer_zoo import build_local_model, list_local_arches

    args = parse_args()
    if not args.list and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/style_transfer_zoo.py --list")
        print("- python scripts/style_transfer_zoo.py --search tiny --list")
        print("- python scripts/style_transfer_zoo.py --smoke dlst:adain_tiny")
        return 2

    arches = list_local_arches()
    if args.search:
        needle = str(args.search).lower()
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Style transfer local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"dlst:{arch_id}"

        import torch

        content = torch.randn(
            int(args.batch_size),
            int(args.in_channels),
            int(args.image_size),
            int(args.image_size),
        )
        style = torch.randn(
            int(args.batch_size),
            int(args.in_channels),
            int(args.image_size),
            int(args.image_size),
        )
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            image_size=int(args.image_size),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
            steps=int(args.steps),
            strength=float(args.strength),
            style_dim=int(args.style_dim),
            control_scale=float(args.control_scale),
            num_layers=int(args.num_layers),
            guidance_scale=float(args.guidance_scale),
            ref_weight=float(args.ref_weight),
            temperature=float(args.temperature),
        )
        model.eval()
        with torch.no_grad():
            out = model(content, style)

        print("")
        print(f"smoke: {arch_id}")
        print(f"- model={type(model).__name__}")
        print(f"- content_shape={tuple(content.shape)}")
        print(f"- style_shape={tuple(style.shape)}")
        print(f"- output={_summarize(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
