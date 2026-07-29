from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


_FAMILIES = [
    "unet_irisseg",
    "deeplab_irisseg",
    "boundary_irisseg",
    "transformer_irisseg",
    "prompt_irisseg",
    "dual_irisseg",
    "coarse_irisseg",
    "contour_irisseg",
    "pupil_irisseg",
    "mamba_irisseg",
]
_SIZES = ("tiny", "small", "base")


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    width_mult: float = 1.0


class UnknownLocalArch(KeyError):
    pass


Builder = Callable[[BuildConfig], object]


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="irisseg")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.vision.iris_segmentation.{family}",
        builder_template="build_{family}_iris_segmentor",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="irisseg")


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"iris_segmentation", "iris_segment"}:
        prefix = "irisseg"
    if prefix not in {"irisseg", "local"}:
        raise ValueError(f"Unsupported iris segmentation prefix: {prefix!r} (arch_id={arch_id!r})")
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown iris segmentation arch: {arch_id!r}. Tip: import `dlhub.vision.iris_segmentation_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
