from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {'contrastive_sign_digit_tiny': {'width': 24, 'depth': 1, 'num_classes': 10}, 'contrastive_sign_digit_small': {'width': 36, 'depth': 2, 'num_classes': 10}, 'contrastive_sign_digit_base': {'width': 48, 'depth': 3, 'num_classes': 10}}


def build_contrastive_sign_digit_sign_digit_classifier(
    *,
    in_channels: int,
    variant: str = 'contrastive_sign_digit_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_classifier(
        family='contrastive_sign_digit',
        mode='contrastive',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(build_contrastive_sign_digit_sign_digit_classifier, 'contrastive_sign_digit_tiny')
