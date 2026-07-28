from __future__ import annotations
from ._common import build_baseline_prompt_learner, smoke_test_prompt_learner

_VARIANTS = {
    "mamba_promptlearn_tiny": {"width": 24, "depth": 1},
    "mamba_promptlearn_small": {"width": 32, "depth": 2},
    "mamba_promptlearn_base": {"width": 48, "depth": 3},
}


def build_mamba_promptlearn_prompt_learner(
    *,
    in_channels: int,
    variant: str = "mamba_promptlearn_small",
    width_mult: float = 1.0,
    prompt_len: int = 8,
):
    return build_baseline_prompt_learner(
        family="mamba_promptlearn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        prompt_len=int(prompt_len),
    )


if __name__ == "__main__":
    smoke_test_prompt_learner(build_mamba_promptlearn_prompt_learner, "mamba_promptlearn_tiny")
