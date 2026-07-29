from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    image_size: int = 32
    vocab_size: int = 128
    seq_len: int = 16
    embed_dim: int = 64
    num_classes: int = 0
    width_mult: float = 1.0
    dropout: float = 0.0


class UnknownLocalArch(KeyError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="vlm")


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
            if name.startswith("build_") and name.endswith("_vlm"):
                return name
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    def _builder(cfg: BuildConfig):
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.multimodal.vlm.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(f"VLM module {module_name!r} missing {builder_name}()")

        kwargs: dict[str, object] = {
            "image_size": int(cfg.image_size),
            "vocab_size": int(cfg.vocab_size),
            "seq_len": int(cfg.seq_len),
            "embed_dim": int(cfg.embed_dim),
            "num_classes": int(cfg.num_classes),
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
    module_dir = here / "vlm"
    if not module_dir.exists():
        return

    for py in sorted(module_dir.glob("*.py")):
        module_name = py.stem
        if module_name == "__init__" or module_name.startswith("_"):
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
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="vlm")


def build_local_model(
    arch_id: str,
    *,
    image_size: int = 32,
    vocab_size: int = 128,
    seq_len: int = 16,
    embed_dim: int = 64,
    num_classes: int = 0,
    width_mult: float = 1.0,
    dropout: float = 0.0,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"mm", "multimodal"}:
        prefix = "vlm"
    if prefix not in {"vlm", "local"}:
        raise ValueError(f"Unsupported VLM prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown VLM arch: {arch_id!r}. Tip: run `python scripts/vlm_zoo.py --list`."
        )
    return builder(
        BuildConfig(
            image_size=int(image_size),
            vocab_size=int(vocab_size),
            seq_len=int(seq_len),
            embed_dim=int(embed_dim),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
