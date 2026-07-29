from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


_FAMILIES = [
    "deep_relight",
    "hdr_relight",
    "intrinsic_relight",
    "ratio_relight",
    "retinex_relight",
    "portrait_relight",
    "transformer_relight",
    "diffusion_relight",
    "prompt_relight",
    "mamba_relight",
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

    return split_arch_id(arch_id, default_prefix="imgrelight")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.vision.image_relighting.{family}",
        builder_template="build_{family}_relighter",
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

    return list_arch_ids(_REGISTRY, prefix="imgrelight")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    width_mult: float = 1.0,
    **builder_kwargs: object,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"relight", "image_relighting"}:
        prefix = "imgrelight"
    if prefix not in {"imgrelight", "local"}:
        raise ValueError(f"Unsupported image relighting prefix: {prefix!r} (arch_id={arch_id!r})")
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown image relighting arch: {arch_id!r}. Tip: import `dlhub.vision.image_relighting_zoo` and call `list_local_arches()`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            width_mult=float(width_mult),
            extras=dict(builder_kwargs),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
