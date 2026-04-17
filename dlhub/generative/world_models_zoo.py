from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int = 3
    action_dim: int = 4
    context_dim: int = 16
    width_mult: float = 1.0


class UnknownLocalArch(KeyError):
    pass


Builder = Callable[[BuildConfig], object]


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "world", arch_id
    prefix_name, name = arch_id.split(":", 1)
    prefix_name = prefix_name.strip().lower()
    name = name.strip()
    if not prefix_name or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix_name, name


def _extract_variants_from_source(src: str) -> list[str] | None:
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        target_name = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_name = target.id
                    break
            value = node.value
        if target_name != "_VARIANTS" or not isinstance(value, ast.Dict):
            continue
        keys = []
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
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
            if name.startswith("build_") and name.endswith("_world_model"):
                return name
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    def _builder(cfg: BuildConfig):
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.generative.world_models.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(f"world-model module {module_name!r} missing {builder_name}()")
        kwargs: dict[str, object] = {
            "in_channels": int(cfg.in_channels),
            "action_dim": int(cfg.action_dim),
            "context_dim": int(cfg.context_dim),
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


def _extend_registry(registry: dict[str, Builder]) -> None:
    module_dir = Path(__file__).resolve().parent / "world_models"
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
        builder_name = _extract_builder_name_from_source(src)
        if not variants or builder_name is None:
            continue
        for variant in variants:
            name = str(variant).lower().strip()
            if name and name not in registry:
                registry[name] = _make_lazy_builder(module_name, builder_name=builder_name, variant=name)


def _registry() -> dict[str, Builder]:
    registry: dict[str, Builder] = {}
    _extend_registry(registry)
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"world:{name}" for name in sorted(_REGISTRY)]


def build_local_model(
    arch_id: str,
    *,
    in_channels: int = 3,
    action_dim: int = 4,
    context_dim: int = 16,
    width_mult: float = 1.0,
):
    prefix_name, name = _split_arch_id(arch_id)
    if prefix_name not in {"world", "local"}:
        raise ValueError(f"Unsupported world-model prefix: {prefix_name!r} (arch_id={arch_id!r})")
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown world-model arch: {arch_id!r}. Tip: import `dlhub.generative.world_models_zoo` and call `list_local_arches()`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            action_dim=int(action_dim),
            context_dim=int(context_dim),
            width_mult=float(width_mult),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
