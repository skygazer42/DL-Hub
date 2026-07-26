from __future__ import annotations

from torch import nn

from ._common import build_toy_face_verifier, smoke_test_face_verifier


_VARIANTS: dict[str, dict[str, int]] = {
    "relation_verify_tiny": {"width": 24, "embedding_dim": 48},
    "relation_verify_small": {"width": 36, "embedding_dim": 64},
    "relation_verify_base": {"width": 48, "embedding_dim": 96},
}


def build_relation_verify_face_verifier(
    *,
    in_channels: int,
    variant: str = "relation_verify_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_verifier(
        family="relation_verify",
        mode="relation",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_verifier(build_relation_verify_face_verifier, "relation_verify_tiny")
