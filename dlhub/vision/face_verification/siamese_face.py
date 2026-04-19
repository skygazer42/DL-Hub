from __future__ import annotations

from torch import nn

from ._common import build_toy_face_verifier, smoke_test_face_verifier


_VARIANTS: dict[str, dict[str, int]] = {'siamese_face_tiny': {'width': 24, 'embedding_dim': 48}, 'siamese_face_small': {'width': 36, 'embedding_dim': 64}, 'siamese_face_base': {'width': 48, 'embedding_dim': 96}}


def build_siamese_face_face_verifier(
    *,
    in_channels: int,
    variant: str = 'siamese_face_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_verifier(
        family='siamese_face',
        mode='siamese',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_verifier(build_siamese_face_face_verifier, 'siamese_face_tiny')
