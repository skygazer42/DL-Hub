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
    rows = list(lines)
    if len(rows) <= int(limit):
        for row in rows:
            print(row)
        return
    head = max(10, int(limit) - 10)
    for row in rows[:head]:
        print(row)
    print(f"... ({len(rows) - int(limit)} more) ...")
    for row in rows[-10:]:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GAN zoo utilities (no downloads).")
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring.")
    parser.add_argument("--limit", type=int, default=80, help="Max lines printed in list mode.")
    parser.add_argument("--timeline", action="store_true", help="Print GAN timeline metadata.")
    parser.add_argument("--list-profiles", action="store_true", help="List recommendation profiles.")
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
    parser.add_argument("--num-classes", type=int, default=10, help="Class count for conditional GANs.")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    from dlhub.generative.gan._recommend import list_profiles, recommend_arches
    from dlhub.generative.gan._timeline import entries
    from dlhub.generative.gan_zoo import build_local_model, list_local_arches

    args = parse_args()
    if (
        not args.list
        and not args.timeline
        and not args.list_profiles
        and args.recommend is None
        and args.smoke is None
    ):
        print("Nothing to do. Try one of:")
        print("- python scripts/gan_zoo.py --list")
        print("- python scripts/gan_zoo.py --timeline")
        print("- python scripts/gan_zoo.py --list-profiles")
        print("- python scripts/gan_zoo.py --recommend balanced --top-k 8")
        print("- python scripts/gan_zoo.py --smoke gan:dcgan_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("GAN local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.timeline:
        timeline = entries()
        if needle:
            timeline = [
                e
                for e in timeline
                if needle in e.family.lower() or needle in e.group.lower() or needle in e.method.lower()
            ]
        print("GAN timeline")
        print(f"- total_families={len(timeline)}")
        print(f"- total_arches={len(arches)}")
        current_year = None
        for e in sorted(timeline, key=lambda x: (9999 if x.year is None else x.year, x.group, x.family)):
            y = "unknown" if e.year is None else str(e.year)
            if y != current_year:
                print("")
                print(y)
                current_year = y
            print(f"- {e.family} [{e.group}]: {e.method} -> gan:{e.family}_tiny")

    if args.list_profiles:
        print("GAN recommendation profiles")
        for p in list_profiles():
            print(f"- {p.key}: {p.title} | {p.summary}")

    if args.recommend is not None:
        try:
            recs = recommend_arches(str(args.recommend), variant=str(args.variant), top_k=int(args.top_k))
        except ValueError as exc:
            print(str(exc))
            print("\nTip: run `python scripts/gan_zoo.py --list-profiles`")
            return 2

        print("GAN recommendations")
        print(f"- profile={str(args.recommend).strip().lower()}")
        print(f"- variant={str(args.variant).strip().lower()}")
        print(f"- top_k={int(args.top_k)}")
        print("")
        for idx, r in enumerate(recs, start=1):
            year = "unknown" if r.year is None else str(r.year)
            print(
                f"{idx:02d}. {r.arch_id} | group={r.group} | year={year} | score={float(r.score):.3f}"
            )
            print(f"    reason: {r.reason}")

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"gan:{arch_id}"
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

