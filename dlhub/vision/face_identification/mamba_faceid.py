from __future__ import annotations

from torch import nn

from ._common import build_toy_face_identifier, smoke_test_face_identifier


_VARIANTS: dict[str, dict[str, int]] = {'mamba_faceid_tiny': {'width': 24, 'depth': 1, 'embedding_dim': 48}, 'mamba_faceid_small': {'width': 36, 'depth': 2, 'embedding_dim': 64}, 'mamba_faceid_base': {'width': 48, 'depth': 3, 'embedding_dim': 96}}


def build_mamba_faceid_face_identifier(
    *,
    in_channels: int,
    variant: str = 'mamba_faceid_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_identifier(
        family='mamba_faceid',
        mode='mamba',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_identifier(build_mamba_faceid_face_identifier, 'mamba_faceid_tiny')
