from __future__ import annotations

from torch import nn

from ._common import build_toy_face_verifier, smoke_test_face_verifier


_VARIANTS: dict[str, dict[str, int]] = {'triplet_face_tiny': {'width': 24, 'embedding_dim': 48}, 'triplet_face_small': {'width': 36, 'embedding_dim': 64}, 'triplet_face_base': {'width': 48, 'embedding_dim': 96}}


def build_triplet_face_face_verifier(
    *,
    in_channels: int,
    variant: str = 'triplet_face_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_verifier(
        family='triplet_face',
        mode='triplet',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_verifier(build_triplet_face_face_verifier, 'triplet_face_tiny')
