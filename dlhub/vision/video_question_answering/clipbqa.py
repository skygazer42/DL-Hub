from __future__ import annotations
from ._common import build_baseline_video_qa, smoke_test_video_qa

_VARIANTS = {
    "clipbqa_tiny": {"width": 24, "depth": 1},
    "clipbqa_small": {"width": 32, "depth": 2},
    "clipbqa_base": {"width": 48, "depth": 3},
}


def build_clipbqa_video_qa_model(
    *,
    in_channels: int,
    variant: str = "clipbqa_small",
    width_mult: float = 1.0,
    answer_vocab: int = 32,
):
    return build_baseline_video_qa(
        family="clipbqa",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        answer_vocab=int(answer_vocab),
    )


if __name__ == "__main__":
    smoke_test_video_qa(build_clipbqa_video_qa_model, "clipbqa_tiny")
