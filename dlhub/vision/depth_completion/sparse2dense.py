from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "sparse2dense_tiny": {"width": 24, "depth": 1},
    "sparse2dense_small": {"width": 32, "depth": 2},
    "sparse2dense_base": {"width": 48, "depth": 3},
}


def build_sparse2dense_depth_completer(
    *, in_channels: int, variant: str = "sparse2dense_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="sparse2dense",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_sparse2dense_depth_completer, "sparse2dense_tiny")
