from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib


_FAMILIES = [
    "prototype_fsseg",
    "matching_fsseg",
    "relation_fsseg",
    "attention_fsseg",
    "transformer_fsseg",
    "prompt_fsseg",
    "hypercorr_fsseg",
    "dual_fsseg",
    "iterative_fsseg",
    "mamba_fsseg",
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
        return "fsseg", arch_id
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
                module = importlib.import_module(f"dlhub.vision.few_shot_segmentation.{family}")
                fn = getattr(module, f"build_{family}_few_shot_segmentor")
                return fn(
                    in_channels=int(cfg.in_channels),
                    variant=str(variant),
                    width_mult=float(cfg.width_mult),
                )

            registry[variant] = _builder
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"fsseg:{name}" for name in sorted(_REGISTRY)]


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"few_shot_segmentation", "fewshot_segmentation", "fewshot_seg"}:
        prefix = "fsseg"
    if prefix not in {"fsseg", "local"}:
        raise ValueError(
            f"Unsupported few shot segmentation prefix: {prefix!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown few shot segmentation arch: {arch_id!r}. Tip: import `dlhub.vision.few_shot_segmentation_zoo` and call `list_local_arches()`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            width_mult=float(width_mult),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
