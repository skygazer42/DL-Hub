from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    num_classes: int
    image_size: int = 64
    width_mult: float = 1.0
    dropout: float = 0.0


class UnknownLocalArch(KeyError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="fparse")


Builder = Callable[[BuildConfig], object]


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
        if isinstance(node, ast.FunctionDef):
            name = str(node.name)
            if name.startswith("build_") and name.endswith("_face_parser"):
                return name
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    module_name = str(module_name).strip()
    builder_name = str(builder_name).strip()
    variant = str(variant).strip()

    def _builder(cfg: BuildConfig):
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.vision.face_parsing.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(f"Face parsing module {module_name!r} missing {builder_name}()")

        kwargs: dict[str, object] = {
            "in_channels": int(cfg.in_channels),
            "num_classes": int(cfg.num_classes),
            "image_size": int(cfg.image_size),
            "variant": str(variant),
            "width_mult": float(cfg.width_mult),
            "dropout": float(cfg.dropout),
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


def _extend_registry(registry: dict[str, Builder]) -> None:
    here = Path(__file__).resolve().parent
    module_dir = here / "face_parsing"
    if not module_dir.exists():
        return

    for py in sorted(module_dir.glob("*.py")):
        module_name = py.stem
        if module_name in {"__init__", "_common"} or module_name.startswith("_"):
            continue
        src = py.read_text(encoding="utf-8-sig")
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
            if name and name not in registry:
                registry[name] = _make_lazy_builder(
                    module_name, builder_name=builder_name, variant=name
                )


def _registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    _extend_registry(r)
    return r


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="fparse")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.0,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"face", "face_parse", "face_parsing"}:
        prefix = "fparse"
    if prefix not in {"fparse", "local"}:
        raise ValueError(f"Unsupported face parsing prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown face parsing arch: {arch_id!r}. Tip: run `python scripts/face_parsing_zoo.py --list`."
        )

    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            image_size=int(image_size),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
