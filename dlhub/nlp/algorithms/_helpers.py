from __future__ import annotations

from collections.abc import Callable

from torch import nn

from dlhub.nlp.types import BuildConfig, Builder


def make_builder(build_fn: Callable[..., nn.Module], *, variant: str) -> Builder:
    v = str(variant)

    def _build(cfg: BuildConfig) -> nn.Module:
        return build_fn(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            max_length=int(cfg.max_length),
            num_classes=int(cfg.num_classes),
            width_mult=float(cfg.width_mult),
            dropout=float(cfg.dropout),
            variant=v,
        )

    return _build

