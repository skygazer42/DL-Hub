from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib


_FAMILIES = [
    "glassseg_toy",
    "translab_seg",
    "refractmask_seg",
    "camotransparent_seg",
    "trimap_transparent",
    "boundary_glass_seg",
    "transformer_transparent",
    "diffusion_transparent",
    "prompt_transparent",
    "mamba_transparent",
]
_SIZES = ("tiny", "small", "base")


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    width_mult: float = 1.0
    extras: dict[str, object] = field(default_factory=dict)


class UnknownLocalArch(KeyError):
    pass


Builder = Callable[[BuildConfig], object]


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "tos", arch_id
    prefix, name = arch_id.split(":", 1)
    prefix = prefix.strip().lower()
    name = name.strip()
    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix, name


def _registry() -> dict[str, Builder]:
    registry: dict[str, Builder] = {}
    for family in _FAMILIES:
        for size in _SIZES:
            variant = f"{family}_{size}"

            def _builder(cfg: BuildConfig, family: str = family, variant: str = variant):
                module = importlib.import_module(
                    f"dlhub.vision.transparent_object_segmentation.{family}"
                )
                fn = getattr(module, f"build_{family}_transparent_segmenter")
                return fn(
                    in_channels=int(cfg.in_channels),
                    variant=str(variant),
                    width_mult=float(cfg.width_mult),
                    **dict(cfg.extras),
                )

            registry[variant] = _builder
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"tos:{name}" for name in sorted(_REGISTRY)]


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    width_mult: float = 1.0,
    **builder_kwargs: object,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"transparent", "transparent_object_segmentation"}:
        prefix = "tos"
    if prefix not in {"tos", "local"}:
        raise ValueError(
            f"Unsupported transparent object segmentation prefix: {prefix!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            "Unknown transparent object segmentation arch: "
            f"{arch_id!r}. Tip: import `dlhub.vision.transparent_object_segmentation_zoo` "
            "and call `list_local_arches()`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            width_mult=float(width_mult),
            extras=dict(builder_kwargs),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
