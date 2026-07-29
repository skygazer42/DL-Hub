from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


_FAMILIES = [
    "coop_prompt",
    "cocoop_prompt",
    "proda_prompt",
    "vpt_prompt",
    "promptsrc_prompt",
    "maple_prompt",
    "dapt_prompt",
    "adapter_prompt",
    "prefix_fusion_prompt",
    "mamba_promptlearn",
]
_SIZES = ("tiny", "small", "base")


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    prompt_len: int = 8
    width_mult: float = 1.0


class UnknownLocalArch(KeyError):
    pass


Builder = Callable[[BuildConfig], object]


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="mpl")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.multimodal.prompt_learning.{family}",
        builder_template="build_{family}_prompt_learner",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
            prompt_len=int(cfg.prompt_len),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="mpl")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    prompt_len: int = 8,
    width_mult: float = 1.0,
):
    prefix, name = _split_arch_id(arch_id)
    if prefix in {"prompt", "prompt_learning"}:
        prefix = "mpl"
    if prefix not in {"mpl", "local"}:
        raise ValueError(f"Unsupported prompt learning prefix: {prefix!r} (arch_id={arch_id!r})")
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown prompt learning arch: {arch_id!r}. Tip: import `dlhub.multimodal.prompt_learning_zoo` and call `list_local_arches()`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            prompt_len=int(prompt_len),
            width_mult=float(width_mult),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
