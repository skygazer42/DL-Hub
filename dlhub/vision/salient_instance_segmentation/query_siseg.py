from __future__ import annotations

from torch import nn

from ._common import build_baseline_salient_instance_segmentor, smoke_test_salient_instance_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "query_siseg_tiny": {"width": 24, "depth": 1, "queries": 8, "protos": 6},
    "query_siseg_small": {"width": 36, "depth": 2, "queries": 12, "protos": 8},
    "query_siseg_base": {"width": 48, "depth": 3, "queries": 16, "protos": 10},
}


def build_query_siseg_salient_instance_segmentor(
    *, in_channels: int, variant: str = "query_siseg_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_salient_instance_segmentor(
        family="query_siseg",
        mode="query",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_salient_instance_segmentor(
        build_query_siseg_salient_instance_segmentor, "query_siseg_tiny"
    )
