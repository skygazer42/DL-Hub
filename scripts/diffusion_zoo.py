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

    print_limited(lines, limit=limit, annotate_fidelity=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local diffusion zoo utilities (no downloads).")
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring.")
    parser.add_argument("--limit", type=int, default=80, help="Max lines printed in list mode.")
    parser.add_argument(
        "--timeline", action="store_true", help="Print diffusion timeline metadata."
    )
    parser.add_argument(
        "--list-profiles", action="store_true", help="List recommendation profiles."
    )
    parser.add_argument(
        "--recommend",
        type=str,
        default=None,
        metavar="PROFILE",
        help="Recommend architectures for a profile.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for --recommend.")
    parser.add_argument("--variant", type=str, default="tiny", help="tiny | small | base")
    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        metavar="ARCH_ID",
        help="Run a short forward smoke for an architecture id.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke.")
    parser.add_argument("--in-channels", type=int, default=3, help="Image channels.")
    parser.add_argument("--image-size", type=int, default=32, help="Image size for smoke.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension.")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Class count for conditional diffusion variants.",
    )
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    from dlhub.generative.diffusion._recommend import list_profiles, recommend_arches
    from dlhub.generative.diffusion._timeline import entries
    from dlhub.generative.diffusion_zoo import build_local_model, list_local_arches

    args = parse_args()
    if (
        not args.list
        and not args.timeline
        and not args.list_profiles
        and args.recommend is None
        and args.smoke is None
    ):
        print("Nothing to do. Try one of:")
        print("- python scripts/diffusion_zoo.py --list")
        print("- python scripts/diffusion_zoo.py --timeline")
        print("- python scripts/diffusion_zoo.py --list-profiles")
        print("- python scripts/diffusion_zoo.py --recommend fidelity --top-k 8")
        print("- python scripts/diffusion_zoo.py --smoke diff:ddpm_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [arch for arch in arches if needle in arch.lower()]

    if args.list:
        print("Diffusion local zoo")
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
                or needle in entry.group.lower()
                or needle in entry.method.lower()
            ]
        print("Diffusion timeline")
        print(f"- total_families={len(timeline)}")
        print(f"- total_arches={len(arches)}")
        current_year = None
        for entry in sorted(
            timeline,
            key=lambda x: (9999 if x.year is None else x.year, x.group, x.family),
        ):
            year = "unknown" if entry.year is None else str(entry.year)
            if year != current_year:
                print("")
                print(year)
                current_year = year
            print(f"- {entry.family} [{entry.group}]: {entry.method} -> diff:{entry.family}_tiny")

    if args.list_profiles:
        print("Diffusion recommendation profiles")
        for profile in list_profiles():
            print(f"- {profile.key}: {profile.title} | {profile.summary}")

    if args.recommend is not None:
        try:
            recs = recommend_arches(
                str(args.recommend),
                variant=str(args.variant),
                top_k=int(args.top_k),
            )
        except ValueError as exc:
            print(str(exc))
            print("\nTip: run `python scripts/diffusion_zoo.py --list-profiles`")
            return 2

        print("Diffusion recommendations")
        print(f"- profile={str(args.recommend).strip().lower()}")
        print(f"- variant={str(args.variant).strip().lower()}")
        print(f"- top_k={int(args.top_k)}")
        print("")
        for idx, rec in enumerate(recs, start=1):
            year = "unknown" if rec.year is None else str(rec.year)
            print(
                f"{idx:02d}. {rec.arch_id} | group={rec.group} | year={year} | score={float(rec.score):.3f}"
            )
            print(f"    reason: {rec.reason}")

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"diff:{arch_id}"
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            image_size=int(args.image_size),
            latent_dim=int(args.latent_dim),
            num_classes=int(args.num_classes),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
        )
        out = model.forward(batch_size=int(args.batch_size))
        print("")
        print(f"smoke: {arch_id}")
        print(f"- output={_summarize(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
