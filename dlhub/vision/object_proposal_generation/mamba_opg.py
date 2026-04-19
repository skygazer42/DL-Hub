from __future__ import annotations

from torch import nn

from ._common import build_toy_proposer, smoke_test_proposer


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_opg_tiny": {"width": 24, "depth": 1, "proposals": 20},
    "mamba_opg_small": {"width": 36, "depth": 2, "proposals": 24},
    "mamba_opg_base": {"width": 48, "depth": 3, "proposals": 32},
}


def build_mamba_opg_proposer(
    *,
    in_channels: int,
    variant: str = "mamba_opg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_proposer(
        family="mamba_opg",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_proposer(build_mamba_opg_proposer, "mamba_opg_tiny")
