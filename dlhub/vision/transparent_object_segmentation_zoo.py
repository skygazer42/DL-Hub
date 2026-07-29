from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


_FAMILIES = [
    "glassseg_baseline",
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
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="tos")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.vision.transparent_object_segmentation.{family}",
        builder_template="build_{family}_transparent_segmenter",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
            **dict(cfg.extras),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="tos")


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
