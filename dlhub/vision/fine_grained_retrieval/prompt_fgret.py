from __future__ import annotations
from torch import nn
from ._common import build_baseline_retrieval_model, smoke_test_retrieval

_VARIANTS = {
    "prompt_fgret_tiny": {"width": 24, "depth": 1, "embed": 128},
    "prompt_fgret_small": {"width": 32, "depth": 2, "embed": 160},
    "prompt_fgret_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_prompt_fgret_(
    *, in_channels: int, variant: str = "prompt_fgret_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_retrieval_model(
        family="prompt_fgret",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_retrieval(build_prompt_fgret_, "prompt_fgret_tiny")
