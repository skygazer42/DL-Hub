from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {'transformer_handedness_tiny': {'width': 24, 'depth': 1, 'num_classes': 2}, 'transformer_handedness_small': {'width': 36, 'depth': 2, 'num_classes': 2}, 'transformer_handedness_base': {'width': 48, 'depth': 3, 'num_classes': 2}}


def build_transformer_handedness_handedness_classifier(
    *,
    in_channels: int,
    variant: str = 'transformer_handedness_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_classifier(
        family='transformer_handedness',
        mode='transformer',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(build_transformer_handedness_handedness_classifier, 'transformer_handedness_tiny')
