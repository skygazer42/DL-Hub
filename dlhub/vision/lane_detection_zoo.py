from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from torch import nn


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    num_lanes: int
    image_size: int | tuple[int, int] = 64
    width_mult: float = 1.0
    dropout: float = 0.0
    num_points: int = 16
    num_rows: int = 16
    grid_size: int = 32
    num_anchors: int = 24
    num_queries: int = 6
    extras: Mapping[str, object] = field(default_factory=dict)

    def to_builder_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "in_channels": int(self.in_channels),
            "num_lanes": int(self.num_lanes),
            "image_size": _normalize_image_size(self.image_size),
            "width_mult": float(self.width_mult),
            "dropout": float(self.dropout),
            "num_points": int(self.num_points),
            "num_rows": int(self.num_rows),
            "grid_size": int(self.grid_size),
            "num_anchors": int(self.num_anchors),
            "num_queries": int(self.num_queries),
        }
        kwargs.update(dict(self.extras))
        return kwargs


class UnknownLocalArch(KeyError):
    pass


def _normalize_image_size(image_size: int | tuple[int, int] | list[int]) -> int | tuple[int, int]:
    if isinstance(image_size, list | tuple):
        if len(image_size) != 2:
            raise ValueError(f"image_size sequence must have length 2, got {image_size!r}")
        return (int(image_size[0]), int(image_size[1]))
    return int(image_size)


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="dllane")


Builder = Callable[[BuildConfig], nn.Module]


def _extract_string_literals(node) -> list[str] | None:
    import ast

    values = None
    if isinstance(node, ast.Dict):
        values = node.keys
    elif isinstance(node, ast.List | ast.Tuple | ast.Set):
        values = node.elts

    if values is None:
        return None

    items: list[str] = []
    for value in values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            items.append(value.value)
    return items or None


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

        if target_name != "_VARIANTS":
            continue

        variants = _extract_string_literals(value)
        if variants:
            return variants

    return None


def _extract_builder_name_from_source(src: str) -> str | None:
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        name = str(node.name)
        if name.startswith("build_") and name.endswith("_lane_detector"):
            return name
    return None


def _filter_kwargs_for_signature(fn, kwargs: dict[str, object]) -> dict[str, object]:
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values()
    ):
        return kwargs

    accepted = {
        name
        for name, parameter in sig.parameters.items()
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    module_name = str(module_name).strip()
    builder_name = str(builder_name).strip()
    variant = str(variant).strip()

    def _builder(cfg: BuildConfig) -> nn.Module:
        import importlib

        mod = importlib.import_module(f"dlhub.vision.lane_detection.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(f"Lane detection module {module_name!r} missing {builder_name}()")

        kwargs = cfg.to_builder_kwargs()
        kwargs["variant"] = str(variant)
        return fn(**_filter_kwargs_for_signature(fn, kwargs))

    return _builder


def _extend_registry_with_discovered_modules(registry: dict[str, Builder]) -> None:
    here = Path(__file__).resolve().parent
    module_dir = here / "lane_detection"

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

        if "_VARIANTS" not in src or "_lane_detector" not in src:
            continue

        variants = _extract_variants_from_source(src)
        if not variants:
            continue

        builder_name = _extract_builder_name_from_source(src)
        if builder_name is None:
            continue

        for variant in variants:
            name = str(variant).lower().strip()
            if not name or name in registry:
                continue
            registry[name] = _make_lazy_builder(
                module_name,
                builder_name=builder_name,
                variant=name,
            )


def _registry() -> dict[str, Builder]:
    registry: dict[str, Builder] = {}
    _extend_registry_with_discovered_modules(registry)
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="dllane")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int | tuple[int, int] = 64,
    width_mult: float = 1.0,
    dropout: float = 0.0,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    **builder_kwargs: object,
) -> nn.Module:
    prefix, name = _split_arch_id(arch_id)
    if prefix == "lane":
        prefix = "dllane"
    if prefix not in {"dllane", "local"}:
        raise ValueError(f"Unsupported lane detection prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown lane detection arch: {arch_id!r}. Tip: run `python scripts/lane_detection_zoo.py --list`."
        )

    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            num_lanes=int(num_lanes),
            image_size=_normalize_image_size(image_size),
            width_mult=float(width_mult),
            dropout=float(dropout),
            num_points=int(num_points),
            num_rows=int(num_rows),
            grid_size=int(grid_size),
            num_anchors=int(num_anchors),
            num_queries=int(num_queries),
            extras=dict(builder_kwargs),
        )
    )


__all__ = [
    "BuildConfig",
    "UnknownLocalArch",
    "build_local_model",
    "list_local_arches",
]
