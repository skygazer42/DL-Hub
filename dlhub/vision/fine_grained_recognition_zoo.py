from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch import nn


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    num_classes: int
    image_size: int = 64
    width_mult: float = 1.0
    dropout: float = 0.1


class UnknownLocalArch(KeyError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="dlfgvc")


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
        if target_name != "_VARIANTS":
            continue
        if isinstance(value, ast.Dict):
            keys: list[str] = []
            for k in value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.append(k.value)
            return keys or None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "make_fgvc_variants"
        ):
            if (
                value.args
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
            ):
                prefix = str(value.args[0].value).strip()
                if prefix:
                    return [f"{prefix}_tiny", f"{prefix}_small", f"{prefix}_base"]
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
        if name.startswith("build_") and name.endswith("_fgvc_classifier"):
            return name
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    def _builder(cfg: BuildConfig) -> nn.Module:
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.vision.fine_grained_recognition.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(f"FGVC module {module_name!r} missing {builder_name}()")

        kwargs: dict[str, object] = {
            "in_channels": int(cfg.in_channels),
            "num_classes": int(cfg.num_classes),
            "variant": str(variant),
            "image_size": int(cfg.image_size),
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


def _extend_registry(r: dict[str, Builder]) -> None:
    here = Path(__file__).resolve().parent
    module_dir = here / "fine_grained_recognition"
    if not module_dir.exists():
        return

    for py in sorted(module_dir.glob("*.py")):
        module_name = py.stem
        if module_name in {"__init__"} or module_name.startswith("_"):
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
    _extend_registry(r)
    return r


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="dlfgvc")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    prefix, name = _split_arch_id(arch_id)
    if prefix == "fgvc":
        prefix = "dlfgvc"
    if prefix not in {"dlfgvc", "local"}:
        raise ValueError(f"Unsupported FGVC prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown FGVC arch: {arch_id!r}. Tip: run `python scripts/fine_grained_recognition_zoo.py --list`."
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
