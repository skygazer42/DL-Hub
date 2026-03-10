from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch import nn


@dataclass(frozen=True)
class BuildConfig:
    # Shared
    in_channels: int
    num_classes: int
    width_mult: float = 1.0
    dropout: float = 0.1

    # Video-only
    image_size: int = 64
    frames: int = 8

    # Skeleton-only
    num_joints: int = 17
    seq_len: int = 32


class UnknownLocalArch(KeyError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "dlactv", arch_id

    prefix, name = arch_id.split(":", 1)
    prefix = prefix.strip().lower()
    name = name.strip()
    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix, name


Builder = Callable[[BuildConfig], nn.Module]


def _extract_variants_from_source(src: str) -> list[str] | None:
    """Extract `_VARIANTS` keys from a module source without importing it."""

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


def _extract_builder_name_from_source(src: str) -> tuple[str, str] | None:
    """Return (builder_name, modality) for a module source.

    The modality is inferred from the builder suffix:
    - build_*_video_classifier  -> "video"
    - build_*_skeleton_classifier -> "skeleton"
    """

    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        name = str(node.name)
        if name.startswith("build_") and name.endswith("_video_classifier"):
            return name, "video"
        if name.startswith("build_") and name.endswith("_skeleton_classifier"):
            return name, "skeleton"
    return None


def _make_lazy_builder(module_name: str, *, builder_name: str, variant: str) -> Builder:
    module_name = str(module_name).strip()
    builder_name = str(builder_name).strip()
    variant = str(variant).strip()

    def _builder(cfg: BuildConfig) -> nn.Module:
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.vision.action_recognition.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(
                f"Action recognition module {module_name!r} missing {builder_name}()"
            )

        kwargs: dict[str, object] = {
            "in_channels": int(cfg.in_channels),
            "num_classes": int(cfg.num_classes),
            "variant": str(variant),
            "width_mult": float(cfg.width_mult),
            "dropout": float(cfg.dropout),
            "image_size": int(cfg.image_size),
            "frames": int(cfg.frames),
            "num_joints": int(cfg.num_joints),
            "seq_len": int(cfg.seq_len),
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


def _extend_registry(r_video: dict[str, Builder], r_skel: dict[str, Builder]) -> None:
    here = Path(__file__).resolve().parent
    module_dir = here / "action_recognition"
    if not module_dir.exists():
        return

    hidden = {"__init__", "_common", "_timeline"}

    for py in sorted(module_dir.glob("*.py")):
        module_name = py.stem
        if module_name in hidden or module_name.startswith("_"):
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

        builder_meta = _extract_builder_name_from_source(src)
        if builder_meta is None:
            continue
        builder_name, modality = builder_meta

        registry = r_video if modality == "video" else r_skel
        for v in variants:
            name = str(v).lower().strip()
            if not name or name in registry:
                continue
            registry[name] = _make_lazy_builder(
                module_name, builder_name=builder_name, variant=name
            )


def _registries() -> tuple[dict[str, Builder], dict[str, Builder]]:
    r_video: dict[str, Builder] = {}
    r_skel: dict[str, Builder] = {}
    _extend_registry(r_video, r_skel)
    return r_video, r_skel


_VIDEO_REGISTRY, _SKELETON_REGISTRY = _registries()


def list_local_arches() -> list[str]:
    """List all available local action recognition architecture ids."""

    out: list[str] = []
    out.extend([f"dlactv:{name}" for name in sorted(_VIDEO_REGISTRY)])
    out.extend([f"dlacts:{name}" for name in sorted(_SKELETON_REGISTRY)])
    return out


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Video
    image_size: int = 64,
    frames: int = 8,
    # Skeleton
    num_joints: int = 17,
    seq_len: int = 32,
) -> nn.Module:
    """Build a local action recognition model by architecture id.

    Prefixes:
    - `dlactv:<variant>`: video action models (NCTHW input)
    - `dlacts:<variant>`: skeleton action models (NCTV input)
    """

    prefix, name = _split_arch_id(arch_id)
    if prefix in {"actv", "video", "action_video"}:
        prefix = "dlactv"
    if prefix in {"acts", "skel", "skeleton", "action_skeleton"}:
        prefix = "dlacts"
    if prefix not in {"dlactv", "dlacts", "local"}:
        raise ValueError(f"Unsupported action recognition prefix: {prefix!r} (arch_id={arch_id!r})")

    key = str(name).lower().strip()
    if prefix == "dlactv":
        builder = _VIDEO_REGISTRY.get(key)
    elif prefix == "dlacts":
        builder = _SKELETON_REGISTRY.get(key)
    else:  # local: try both
        builder = _VIDEO_REGISTRY.get(key) or _SKELETON_REGISTRY.get(key)

    if builder is None:
        raise UnknownLocalArch(
            f"Unknown action arch: {arch_id!r}. Tip: run `python scripts/action_recognition_zoo.py --list`."
        )

    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            image_size=int(image_size),
            frames=int(frames),
            num_joints=int(num_joints),
            seq_len=int(seq_len),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
