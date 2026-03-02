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
    if hasattr(obj, "logits"):
        return f"{type(obj).__name__}(logits={_summarize(getattr(obj, 'logits'))})"
    return f"{type(obj).__name__}"


def _print_lines(lines: Iterable[str], *, limit: int = 60) -> None:
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


def _dummy_inputs(prefix: str, *, batch_size: int, image_size: int, time: int):
    import torch

    prefix = str(prefix).lower().strip()
    b = int(batch_size)
    s = int(image_size)
    t = int(time)

    if prefix in {"tv", "tvq", "timm", "tvseg", "dl", "local"}:
        return (torch.randn(b, 3, s, s),)
    if prefix == "tvdet":
        # Detection models in torchvision expect a list[Tensor(C, H, W)].
        return ([torch.rand(3, s, s) for _ in range(b)],)
    if prefix == "tvflow":
        # Optical flow models expect (image1, image2) tensors.
        return (torch.rand(b, 3, s, s), torch.rand(b, 3, s, s))
    if prefix == "tvvideo":
        return (torch.rand(b, 3, t, s, s),)

    raise ValueError(f"Unknown prefix for dummy inputs: {prefix!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision model zoo (torchvision/timm) utilities.")

    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring (case-insensitive).")
    parser.add_argument("--limit", type=int, default=60, help="Max lines to print when listing.")

    parser.add_argument("--smoke", type=str, default=None, metavar="ARCH_ID", help="Run a forward smoke on an arch id.")
    parser.add_argument("--num-classes", type=int, default=None, help="Optional num_classes override when building.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke inputs (where applicable).")
    parser.add_argument("--image-size", type=int, default=64, help="Image size for smoke inputs.")
    parser.add_argument("--time", type=int, default=8, help="Video time dimension for tvvideo smoke inputs.")

    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()

    from dlhub.vision.zoo import (
        DependencyNotAvailable,
        build_timm_model,
        build_torchvision_model,
        list_vision_arches,
    )

    args = parse_args()

    if not args.list and args.smoke is None:
        print("Nothing to do. Try one of:")
        print("- python scripts/vision_zoo.py --list")
        print("- python scripts/vision_zoo.py --smoke tv:resnet18")
        return 2

    try:
        arches = list_vision_arches()
    except DependencyNotAvailable as exc:
        print(exc)
        return 1

    if args.search:
        needle = str(args.search).lower()
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        counts: dict[str, int] = {}
        for arch in arches:
            prefix = arch.split(":", 1)[0]
            counts[prefix] = counts.get(prefix, 0) + 1

        print("Vision zoo")
        print(f"- total_arches={len(arches)}")
        for k in sorted(counts):
            print(f"- {k}={counts[k]}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.smoke is not None:
        arch_id = str(args.smoke).strip()
        prefix = arch_id.split(":", 1)[0].lower()

        import torch

        if prefix in {"dl", "local"}:
            from dlhub.vision.local_zoo import build_local_model

            num_classes = int(args.num_classes) if args.num_classes is not None else 1000
            model = build_local_model(
                arch_id,
                in_channels=3,
                num_classes=num_classes,
                image_size=int(args.image_size),
            )
        elif prefix == "timm":
            model = build_timm_model(arch_id, num_classes=args.num_classes)
        else:
            model = build_torchvision_model(arch_id, num_classes=args.num_classes)
        model.eval()

        inputs = _dummy_inputs(
            prefix,
            batch_size=int(args.batch_size),
            image_size=int(args.image_size),
            time=int(args.time),
        )
        with torch.no_grad():
            out = model(*inputs)
        print("")
        print(f"smoke: {arch_id}")
        print(f"- output={_summarize(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
