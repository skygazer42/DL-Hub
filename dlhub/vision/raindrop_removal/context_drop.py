from __future__ import annotations

from torch import nn

from ._common import build_baseline_drop_remover, smoke_test_drop_remover


_VARIANTS: dict[str, dict[str, int]] = {
    "context_drop_tiny": {"width": 24, "depth": 1, "steps": 1},
    "context_drop_small": {"width": 32, "depth": 2, "steps": 2},
    "context_drop_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_context_drop_drop_remover(
    *,
    in_channels: int,
    variant: str = "context_drop_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_drop_remover(
        family="context_drop",
        mode="context",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_drop_remover(build_context_drop_drop_remover, "context_drop_tiny")
