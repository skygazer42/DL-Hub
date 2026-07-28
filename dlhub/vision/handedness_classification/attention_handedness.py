from __future__ import annotations

from torch import nn

from ._common import build_baseline_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {
    "attention_handedness_tiny": {"width": 24, "depth": 1, "num_classes": 2},
    "attention_handedness_small": {"width": 36, "depth": 2, "num_classes": 2},
    "attention_handedness_base": {"width": 48, "depth": 3, "num_classes": 2},
}


def build_attention_handedness_handedness_classifier(
    *,
    in_channels: int,
    variant: str = "attention_handedness_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_hand_classifier(
        family="attention_handedness",
        mode="attention",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(
        build_attention_handedness_handedness_classifier, "attention_handedness_tiny"
    )
