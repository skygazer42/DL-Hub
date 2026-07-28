from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "lxmert_ve_tiny": {"width": 24, "depth": 1},
    "lxmert_ve_small": {"width": 32, "depth": 2},
    "lxmert_ve_base": {"width": 48, "depth": 3},
}


def build_lxmert_ve_visual_entailment_model(
    *, in_channels: int, variant: str = "lxmert_ve_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="lxmert_ve",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_lxmert_ve_visual_entailment_model, "lxmert_ve_tiny")
