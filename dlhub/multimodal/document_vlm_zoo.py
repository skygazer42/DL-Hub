from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass

_FAMILIES = [
    "layoutlm_doc",
    "docformer_doc",
    "pix2struct_doc",
    "donut_doc",
    "table_doc",
    "chart_doc",
    "transformer_doc",
    "retrieval_doc",
    "prompt_doc",
    "mamba_doc",
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

    return split_arch_id(arch_id, default_prefix="docvlm")


def _registry() -> dict[str, Builder]:
    from dlhub.zoo_registry import make_lazy_family_registry

    return make_lazy_family_registry(
        _FAMILIES,
        _SIZES,
        module_template="dlhub.multimodal.document_vlm.{family}",
        builder_template="build_{family}_document_vlm",
        kwargs_factory=lambda cfg, variant: dict(
            in_channels=int(cfg.in_channels),
            variant=str(variant),
            width_mult=float(cfg.width_mult),
        ),
    )


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    from dlhub.zoo_registry import list_arch_ids

    return list_arch_ids(_REGISTRY, prefix="docvlm")


def build_local_model(arch_id: str, *, in_channels: int, width_mult: float = 1.0):
    prefix_name, name = _split_arch_id(arch_id)
    if prefix_name not in {"docvlm", "local"}:
        raise ValueError(f"Unsupported document VLM prefix: {prefix_name!r} (arch_id={arch_id!r})")
    builder = _REGISTRY.get(str(name).lower().strip())
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown document VLM arch: {arch_id!r}. Tip: import `dlhub.multimodal.document_vlm_zoo` and call `list_local_arches()`."
        )
    return builder(BuildConfig(in_channels=int(in_channels), width_mult=float(width_mult)))


__all__ = ["BuildConfig", "UnknownLocalArch", "build_local_model", "list_local_arches"]
