from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib

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
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "eqa", arch_id
    prefix_name, name = arch_id.split(":", 1)
    prefix_name = prefix_name.strip().lower()
    name = name.strip()
    if not prefix_name or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix_name, name


def _registry() -> dict[str, Builder]:
    registry: dict[str, Builder] = {}
    for family in _FAMILIES:
        for size in _SIZES:
            variant = f"{family}_{size}"

            def _builder(cfg: BuildConfig, family: str = family, variant: str = variant):
                module = importlib.import_module(f"dlhub.multimodal.embodied_question_answering.{family}")
                fn = getattr(module, f"build_{family}_embodied_qa_model")
                return fn(
                    in_channels=int(cfg.in_channels),
                    variant=str(variant),
                    width_mult=float(cfg.width_mult),
                    question_dim=int(cfg.question_dim),
                    num_answers=int(cfg.num_answers),
                )

            registry[variant] = _builder
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"eqa:{name}" for name in sorted(_REGISTRY)]


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
