from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

_FAMILIES = [
    "navqa_embodied",
    "memory_eqa",
    "objectnav_eqa",
    "mapqa_embodied",
    "speaker_eqa",
    "transformer_eqa",
    "grounded_eqa",
    "retrieval_eqa",
    "prompt_eqa",
    "mamba_eqa",
]
_SIZES = ("tiny", "small", "base")


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    question_dim: int = 32
    num_answers: int = 8
    width_mult: float = 1.0


class UnknownLocalArch(KeyError):
    pass


Builder = Callable[[BuildConfig], object]


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="eqa")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.multimodal.embodied_question_answering.{family}",
        builder_template="build_{family}_embodied_qa_model",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
            question_dim=int(cfg.question_dim),
            num_answers=int(cfg.num_answers),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="eqa")


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    question_dim: int = 32,
    num_answers: int = 8,
    width_mult: float = 1.0,
):
    prefix_name, name = _split_arch_id(arch_id)
    if prefix_name not in {"eqa", "local"}:
        raise ValueError(f"Unsupported embodied QA prefix: {prefix_name!r} (arch_id={arch_id!r})")
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown embodied QA arch: {arch_id!r}. Tip: import `dlhub.multimodal.embodied_question_answering_zoo` and call `list_local_arches()`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            question_dim=int(question_dim),
            num_answers=int(num_answers),
            width_mult=float(width_mult),
        )
    )


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
