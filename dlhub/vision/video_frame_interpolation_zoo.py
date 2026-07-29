from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


_FAMILIES = [
    "sepconv_interp",
    "super_slomo",
    "dain_baseline",
    "rife_baseline",
    "flavr_baseline",
    "vfi_former",
    "amt_interp",
    "ifrnet_baseline",
    "ema_vfi",
    "mamba_vfi",
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

    return split_arch_id(arch_id, default_prefix="vfi")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.vision.video_frame_interpolation.{family}",
        builder_template="build_{family}_interpolator",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="vfi")


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"video_interp", "interp"}:
        prefix = "vfi"
    if prefix not in {"vfi", "local"}:
        raise ValueError(
            f"Unsupported video frame interpolation prefix: {prefix!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown video frame interpolation arch: {arch_id!r}. Tip: import `dlhub.vision.video_frame_interpolation_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
