from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


_FAMILIES = [
    "vqa_reasoner",
    "chain_reasoner",
    "tool_reasoner",
    "memory_reasoner",
    "grounded_reasoner",
    "program_reasoner",
    "transformer_reasoner",
    "retrieval_reasoner",
    "prompt_reasoner",
    "mamba_reasoner",
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

    return split_arch_id(arch_id, default_prefix="mmr")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.multimodal.multimodal_reasoning.{family}",
        builder_template="build_{family}_reasoner",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="mmr")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    width_mult: float = 1.0,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"reason", "reasoning"}:
        prefix = "mmr"
    if prefix not in {"mmr", "local"}:
        raise ValueError(
            f"Unsupported multimodal reasoning prefix: {prefix!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown multimodal reasoning arch: {arch_id!r}. Tip: import `dlhub.multimodal.multimodal_reasoning_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
