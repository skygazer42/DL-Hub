from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib


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
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "imgrelight", arch_id
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
                module = importlib.import_module(f"dlhub.vision.image_relighting.{family}")
                fn = getattr(module, f"build_{family}_relighter")
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
    return [f"imgrelight:{name}" for name in sorted(_REGISTRY)]


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
