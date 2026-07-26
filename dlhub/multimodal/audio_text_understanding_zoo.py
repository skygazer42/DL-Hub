from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib

_FAMILIES = [
    "audio_bert_understanding",
    "wav2text_understanding",
    "contrastive_atu",
    "event_audio_text",
    "speech_audio_text",
    "transformer_atu",
    "retrieval_atu",
    "diffusion_atu",
    "prompt_atu",
    "mamba_atu",
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
        return "atu", arch_id
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
                module = importlib.import_module(
                    f"dlhub.multimodal.audio_text_understanding.{family}"
                )
                fn = getattr(module, f"build_{family}_audio_text_model")
                return fn(
                    in_channels=int(cfg.in_channels),
                    variant=str(variant),
                    width_mult=float(cfg.width_mult),
                )

            registry[variant] = _builder
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"atu:{name}" for name in sorted(_REGISTRY)]


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix_name, name = _split_arch_id(arch_id)
    if prefix_name not in {"atu", "local"}:
        raise ValueError(
            f"Unsupported audio-text understanding prefix: {prefix_name!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown audio-text understanding arch: {arch_id!r}. Tip: import `dlhub.multimodal.audio_text_understanding_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
