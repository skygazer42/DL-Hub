from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


_FAMILIES = [
    "cnn_thumb_contact",
    "region_thumb_contact",
    "attention_thumb_contact",
    "skeleton_thumb_contact",
    "transformer_thumb_contact",
    "prompt_thumb_contact",
    "contrastive_thumb_contact",
    "graph_thumb_contact",
    "efficient_thumb_contact",
    "mamba_thumb_contact",
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

    return split_arch_id(arch_id, default_prefix="thumbcontact")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.vision.thumb_contact_classification.{family}",
        builder_template="build_{family}_thumb_contact_classifier",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="thumbcontact")


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"thumb_contact_classification", "thumb_contact"}:
        prefix = "thumbcontact"
    if prefix not in {"thumbcontact", "local"}:
        raise ValueError(
            f"Unsupported thumb contact classification prefix: {prefix!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown thumb contact classification arch: {arch_id!r}. Tip: import `dlhub.vision.thumb_contact_classification_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
