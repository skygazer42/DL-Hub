from __future__ import annotations

from torch import nn

from ._common import build_toy_face_identifier, smoke_test_face_identifier


_VARIANTS: dict[str, dict[str, int]] = {
    "proxy_faceid_tiny": {"width": 24, "depth": 1, "embedding_dim": 48},
    "proxy_faceid_small": {"width": 36, "depth": 2, "embedding_dim": 64},
    "proxy_faceid_base": {"width": 48, "depth": 3, "embedding_dim": 96},
}


def build_proxy_faceid_face_identifier(
    *,
    in_channels: int,
    variant: str = "proxy_faceid_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_identifier(
        family="proxy_faceid",
        mode="proxy",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_identifier(build_proxy_faceid_face_identifier, "proxy_faceid_tiny")
