from __future__ import annotations

from torch import nn

from ._common import build_baseline_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {
    "efficient_thumb_contact_tiny": {"width": 24, "depth": 1, "num_classes": 2},
    "efficient_thumb_contact_small": {"width": 36, "depth": 2, "num_classes": 2},
    "efficient_thumb_contact_base": {"width": 48, "depth": 3, "num_classes": 2},
}


def build_efficient_thumb_contact_thumb_contact_classifier(
    *,
    in_channels: int,
    variant: str = "efficient_thumb_contact_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_hand_classifier(
        family="efficient_thumb_contact",
        mode="efficient",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(
        build_efficient_thumb_contact_thumb_contact_classifier, "efficient_thumb_contact_tiny"
    )
