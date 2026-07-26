from __future__ import annotations

from torch import nn

from ._common import build_toy_face_identifier, smoke_test_face_identifier


_VARIANTS: dict[str, dict[str, int]] = {
    "multi_branch_faceid_tiny": {"width": 24, "depth": 1, "embedding_dim": 48},
    "multi_branch_faceid_small": {"width": 36, "depth": 2, "embedding_dim": 64},
    "multi_branch_faceid_base": {"width": 48, "depth": 3, "embedding_dim": 96},
}


def build_multi_branch_faceid_face_identifier(
    *,
    in_channels: int,
    variant: str = "multi_branch_faceid_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_identifier(
        family="multi_branch_faceid",
        mode="multi_branch",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_identifier(
        build_multi_branch_faceid_face_identifier, "multi_branch_faceid_tiny"
    )
