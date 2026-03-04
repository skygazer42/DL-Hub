from __future__ import annotations

import argparse
import importlib
import pkgutil
from types import ModuleType


def list_backbone_modules(*, include_aggregators: bool = False) -> list[str]:
    """List backbone modules under `dlhub.vision.backbones`.

    By default, this hides aggregator/legacy modules like `cnn.py`, because the
    user-facing goal is “one algorithm-family per file”.
    """

    from . import __path__ as backbones_path

    hidden = {"__init__", "catalog", "_blocks", "_transformer"}
    aggregators = {"cnn", "extra_cnn", "transformers", "mixers", "hybrids"}

    names: list[str] = []
    for m in pkgutil.iter_modules(backbones_path):
        name = str(m.name)
        if name in hidden:
            continue
        if name.startswith("_"):
            continue
        if not include_aggregators and name in aggregators:
            continue
        names.append(name)
    return sorted(names)


def import_backbone_module(module_name: str) -> ModuleType:
    name = str(module_name).strip()
    if not name:
        raise ValueError("module_name must be non-empty")
    return importlib.import_module(f"{__package__}.{name}")


def smoke_one(module_name: str, *, image_size: int = 64) -> str:
    """Run a minimal forward on `build_{module_name}_classifier` if present."""

    mod = import_backbone_module(module_name)
    builder_name = f"build_{module_name}_classifier"
    builder = getattr(mod, builder_name, None)
    if builder is None:
        return f"{module_name}: (no {builder_name} found)"

    import torch

    # Prefer defaults per-module: some algorithms use `variant='resnet18'` etc.
    # Some transformer-ish builders accept `image_size`; plain CNNs usually don't.
    try:
        model = builder(in_channels=3, num_classes=10, image_size=int(image_size))
    except TypeError:
        model = builder(in_channels=3, num_classes=10)
    model.eval()
    x = torch.randn(2, 3, int(image_size), int(image_size))
    with torch.no_grad():
        y = model(x)
    shape = tuple(y[0].shape) if isinstance(y, tuple) else tuple(y.shape)
    return f"{module_name}: ok output={shape}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List/smoke local backbone modules (algorithm-family files).")
    p.add_argument("--list", action="store_true", help="List available algorithm-family modules.")
    p.add_argument("--include-aggregators", action="store_true", help="Include legacy aggregator modules.")
    p.add_argument("--smoke", type=str, default=None, help="Smoke one module by name (e.g. resnet).")
    p.add_argument("--image-size", type=int, default=64, help="Image size for smoke.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        mods = list_backbone_modules(include_aggregators=bool(args.include_aggregators))
        print(f"backbones={len(mods)}")
        for m in mods:
            print(m)

    if args.smoke:
        print(smoke_one(str(args.smoke), image_size=int(args.image_size)))

    if not args.list and not args.smoke:
        print("Nothing to do. Try:")
        print("- python -m dlhub.vision.backbones.catalog --list")
        print("- python -m dlhub.vision.backbones.catalog --smoke resnet")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
