from __future__ import annotations

from torch import nn

from ._common import build_baseline_eqa, smoke_test_eqa

_VARIANTS: dict[str, dict[str, int]] = {
    "mapqa_embodied_tiny": {"width": 24, "depth": 1},
    "mapqa_embodied_small": {"width": 32, "depth": 2},
    "mapqa_embodied_base": {"width": 48, "depth": 3},
}


def build_mapqa_embodied_embodied_qa_model(
    *,
    in_channels: int = 3,
    variant: str = "mapqa_embodied_small",
    width_mult: float = 1.0,
    question_dim: int = 32,
    num_answers: int = 8,
) -> nn.Module:
    return build_baseline_eqa(
        family="mapqa_embodied",
        mode="mapqa",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        question_dim=int(question_dim),
        num_answers=int(num_answers),
    )


if __name__ == "__main__":
    smoke_test_eqa(build_mapqa_embodied_embodied_qa_model, "mapqa_embodied_tiny")
