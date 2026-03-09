from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    param_dim: int
    num_clients: int
    local_steps: int
    width_mult: float = 1.0


class UnknownLocalArch(KeyError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "dlfed", arch_id

    prefix, name = arch_id.split(":", 1)
    prefix = prefix.strip().lower()
    name = name.strip()
    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix, name


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
        if not isinstance(node, ast.FunctionDef):
            continue
        name = str(node.name)
        if name.startswith("build_") and name.endswith("_strategy"):
            return name
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    def _builder(cfg: BuildConfig):
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.federated.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(f"Federated module {module_name!r} missing {builder_name}()")

        kwargs: dict[str, object] = {
            "param_dim": int(cfg.param_dim),
            "num_clients": int(cfg.num_clients),
            "local_steps": int(cfg.local_steps),
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


def _extend_registry(r: dict[str, Builder]) -> None:
    here = Path(__file__).resolve().parent
    module_dir = here / "federated"
    if not module_dir.exists():
        return

    for py in sorted(module_dir.glob("*.py")):
        module_name = py.stem
        if module_name in {"__init__"} or module_name.startswith("_"):
            continue
        try:
            src = py.read_text(encoding="utf-8")
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
    return [f"dlfed:{name}" for name in sorted(_REGISTRY)]


def build_local_strategy(
    arch_id: str,
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    width_mult: float = 1.0,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"fed", "fl"}:
        prefix = "dlfed"
    if prefix not in {"dlfed", "local"}:
        raise ValueError(f"Unsupported federated prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown federated arch: {arch_id!r}. Tip: run `python scripts/federated_zoo.py --list`."
        )
    return builder(
        BuildConfig(
            param_dim=int(param_dim),
            num_clients=int(num_clients),
            local_steps=int(local_steps),
            width_mult=float(width_mult),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_strategy", "list_local_arches"]
