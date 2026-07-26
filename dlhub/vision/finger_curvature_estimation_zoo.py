from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib


_FAMILIES = [
    "direct_finger_curvature",
    "heatmap_finger_curvature",
    "graph_finger_curvature",
    "transformer_finger_curvature",
    "prompt_finger_curvature",
    "coarse_to_fine_finger_curvature",
    "multiview_finger_curvature",
    "region_finger_curvature",
    "uncertainty_finger_curvature",
    "mamba_finger_curvature",
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
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "fcurv", arch_id
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
                    f"dlhub.vision.finger_curvature_estimation.{family}"
                )
                fn = getattr(module, f"build_{family}_finger_curvature_estimator")
                return fn(
                    in_channels=int(cfg.in_channels),
                    variant=str(variant),
                    width_mult=float(cfg.width_mult),
                )

            registry[variant] = _builder
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"fcurv:{name}" for name in sorted(_REGISTRY)]


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"finger_curvature_estimation", "finger_curvature"}:
        prefix = "fcurv"
    if prefix not in {"fcurv", "local"}:
        raise ValueError(
            f"Unsupported finger curvature estimation prefix: {prefix!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown finger curvature estimation arch: {arch_id!r}. Tip: import `dlhub.vision.finger_curvature_estimation_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
