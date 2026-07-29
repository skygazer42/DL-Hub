from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch import nn


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    num_classes: int
    width_mult: float = 1.0


class UnknownLocalArch(KeyError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="dlinst")


Builder = Callable[[BuildConfig], nn.Module]


def _extract_variants_from_source(src: str) -> list[str] | None:
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in tree.body:
        target_name: str | None = None
        value = None

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target_name = t.id
                    break
            value = node.value

        if target_name != "_VARIANTS" or not isinstance(value, ast.Dict):
            continue

        keys: list[str] = []
        for k in value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
        return keys or None

    return None


def _extract_builder_name_from_source(src: str) -> str | None:
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        name = str(node.name)
        if name.startswith("build_") and name.endswith("_instance_segmenter"):
            return name
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    module_name = str(module_name).strip()
    builder_name = str(builder_name).strip()
    variant = str(variant).strip()

    def _builder(cfg: BuildConfig) -> nn.Module:
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.vision.instance_segmentation.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(
                f"Instance segmentation module {module_name!r} missing {builder_name}()"
            )

        kwargs: dict[str, object] = {
            "in_channels": int(cfg.in_channels),
            "num_classes": int(cfg.num_classes),
            "variant": str(variant),
            "width_mult": float(cfg.width_mult),
        }

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            params = set(sig.parameters)
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        return fn(**kwargs)

    return _builder


def _extend_registry_with_discovered_modules(r: dict[str, Builder]) -> None:
    here = Path(__file__).resolve().parent
    module_dir = here / "instance_segmentation"

    if not module_dir.exists():
        return

    hidden = {"__init__"}

    for py in sorted(module_dir.glob("*.py")):
        module_name = py.stem
        if module_name in hidden or module_name.startswith("_"):
            continue

        try:
            src = py.read_text(encoding="utf-8-sig")
        except OSError:
            continue

        if "_VARIANTS" not in src or "def build_" not in src:
            continue

        variants = _extract_variants_from_source(src)
        if not variants:
            continue

        builder_name = _extract_builder_name_from_source(src)
        if builder_name is None:
            continue

        for v in variants:
            name = str(v).lower().strip()
            if not name or name in r:
                continue
            r[name] = _make_lazy_builder(module_name, builder_name=builder_name, variant=name)


def _registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    _extend_registry_with_discovered_modules(r)
    return r


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="dlinst")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    num_classes: int,
    width_mult: float = 1.0,
) -> nn.Module:
    prefix, name = _split_arch_id(arch_id)
    if prefix == "inst":
        prefix = "dlinst"
    if prefix not in {"dlinst", "local"}:
        raise ValueError(
            f"Unsupported instance segmentation prefix: {prefix!r} (arch_id={arch_id!r})"
        )

    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown instance segmentation arch: {arch_id!r}. Tip: run `python scripts/instance_segmentation_zoo.py --list`."
        )

    return builder(
        BuildConfig(
            in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(width_mult)
        )
    )


__all__ = [
    "BuildConfig",
    "UnknownLocalArch",
    "build_local_model",
    "list_local_arches",
]
