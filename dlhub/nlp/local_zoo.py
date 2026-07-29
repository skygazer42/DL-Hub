from torch import nn

from dlhub.nlp.algorithms.registry import REGISTRY as _REGISTRY
from dlhub.nlp.types import BuildConfig


class UnknownLocalArch(KeyError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="nl")


def list_local_arches() -> list[str]:
    """List all available local NLP architecture ids (e.g. `nl:bert_tiny`)."""
    return [f"nl:{name}" for name in sorted(_REGISTRY)]


def build_local_model(
    arch_id: str,
    *,
    vocab_size: int,
    pad_id: int,
    max_length: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    """Build a local NLP model by architecture id.

    Architectures are registered as a ResNet-style zoo: one algorithm-family module
    contains multiple variants, and `arch_id` selects a variant by name.
    """

    prefix, name = _split_arch_id(arch_id)
    if prefix not in {"nl", "local"}:
        raise ValueError(f"Unsupported NLP prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(name)
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown NLP arch: {arch_id!r}. Tip: run `python scripts/nlp_zoo.py --list`."
        )
    return builder(
        BuildConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    )


__all__ = [
    "BuildConfig",
    "UnknownLocalArch",
    "build_local_model",
    "list_local_arches",
]
