from __future__ import annotations

from torch import nn

from ._common import build_baseline_face_identifier, smoke_test_face_identifier


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_faceid_tiny": {"width": 24, "depth": 1, "embedding_dim": 48},
    "prompt_faceid_small": {"width": 36, "depth": 2, "embedding_dim": 64},
    "prompt_faceid_base": {"width": 48, "depth": 3, "embedding_dim": 96},
}


def build_prompt_faceid_face_identifier(
    *,
    in_channels: int,
    variant: str = "prompt_faceid_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_face_identifier(
        family="prompt_faceid",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_identifier(build_prompt_faceid_face_identifier, "prompt_faceid_tiny")
