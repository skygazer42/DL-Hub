from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    width_mult: float = 1.0
    num_output_points: int | None = None


class UnknownLocalArch(KeyError):
    pass


Builder = Callable[[BuildConfig], object]


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "pccomp", arch_id

    prefix, name = arch_id.split(":", 1)
    prefix = prefix.strip().lower()
    name = name.strip()
    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix, name


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
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_name = target.id
                    break
            value = node.value
        if target_name != "_VARIANTS" or not isinstance(value, ast.Dict):
            continue
        keys: list[str] = []
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
            if name.startswith("build_") and name.endswith("_completer"):
                return name
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    def _builder(cfg: BuildConfig):
        import importlib
        import inspect

        module = importlib.import_module(f"dlhub.pointcloud.pointcloud_completion.{module_name}")
        fn = getattr(module, builder_name, None)
        if fn is None:
            raise RuntimeError(
                f"Point cloud completion module {module_name!r} missing {builder_name}()"
            )

        kwargs: dict[str, object] = {
            "in_channels": int(cfg.in_channels),
            "variant": str(variant),
            "width_mult": float(cfg.width_mult),
            "num_output_points": cfg.num_output_points,
        }
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            params = set(sig.parameters)
            kwargs = {
                key: value for key, value in kwargs.items() if key in params and value is not None
            }
        return fn(**kwargs)

    return _builder


def _extend_registry(registry: dict[str, Builder]) -> None:
    here = Path(__file__).resolve().parent
    module_dir = here / "pointcloud_completion"
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
        for variant in variants:
            name = str(variant).lower().strip()
            if name and name not in registry:
                registry[name] = _make_lazy_builder(
                    module_name,
                    builder_name=builder_name,
                    variant=name,
                )


def _registry() -> dict[str, Builder]:
    registry: dict[str, Builder] = {}
    _extend_registry(registry)
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"pccomp:{name}" for name in sorted(_REGISTRY)]


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    width_mult: float = 1.0,
    num_output_points: int | None = None,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix not in {"pccomp", "local"}:
        raise ValueError(
            f"Unsupported point cloud completion prefix: {prefix!r} (arch_id={arch_id!r})"
        )

    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            "Unknown point cloud completion arch: "
            f"{arch_id!r}. Tip: import `dlhub.pointcloud.pointcloud_completion_zoo` "
            "and call `list_local_arches()`."
        )

    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            width_mult=float(width_mult),
            num_output_points=num_output_points,
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
