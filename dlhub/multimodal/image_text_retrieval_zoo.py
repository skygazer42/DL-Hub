from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import importlib

_FAMILIES = [
    "clip_retrieval",
    "albef_retrieval",
    "blip_retrieval",
    "dual_encoder_retrieval",
    "cross_attention_retrieval",
    "region_text_retrieval",
    "transformer_retrieval",
    "prompt_retrieval",
    "diffusion_retrieval",
    "mamba_retrieval",
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
        return "itr", arch_id
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
                module = importlib.import_module(f"dlhub.multimodal.image_text_retrieval.{family}")
                fn = getattr(module, f"build_{family}_retriever")
                return fn(
                    in_channels=int(cfg.in_channels),
                    variant=str(variant),
                    width_mult=float(cfg.width_mult),
                )

            registry[variant] = _builder
    return registry


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    return [f"itr:{name}" for name in sorted(_REGISTRY)]


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix_name, name = _split_arch_id(arch_id)
    if prefix_name not in {"itr", "local"}:
        raise ValueError(
            f"Unsupported image-text retrieval prefix: {prefix_name!r} (arch_id={arch_id!r})"
        )
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown image-text retrieval arch: {arch_id!r}. Tip: import `dlhub.multimodal.image_text_retrieval_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
