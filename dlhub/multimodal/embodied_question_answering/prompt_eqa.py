from __future__ import annotations

from torch import nn

from ._common import build_toy_eqa, smoke_test_eqa

_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_eqa_tiny": {"width": 24, "depth": 1},
    "prompt_eqa_small": {"width": 32, "depth": 2},
    "prompt_eqa_base": {"width": 48, "depth": 3},
}


def build_prompt_eqa_embodied_qa_model(
    *,
    in_channels: int = 3,
    variant: str = "prompt_eqa_small",
    width_mult: float = 1.0,
    question_dim: int = 32,
    num_answers: int = 8,
) -> nn.Module:
    return build_toy_eqa(
        family="prompt_eqa",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        question_dim=int(question_dim),
        num_answers=int(num_answers),
    )


if __name__ == "__main__":
    smoke_test_eqa(build_prompt_eqa_embodied_qa_model, "prompt_eqa_tiny")
