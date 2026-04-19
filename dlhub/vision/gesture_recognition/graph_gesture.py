from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {'graph_gesture_tiny': {'width': 24, 'depth': 1, 'num_classes': 4}, 'graph_gesture_small': {'width': 36, 'depth': 2, 'num_classes': 4}, 'graph_gesture_base': {'width': 48, 'depth': 3, 'num_classes': 4}}


def build_graph_gesture_gesture_recognizer(
    *,
    in_channels: int,
    variant: str = 'graph_gesture_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_classifier(
        family='graph_gesture',
        mode='graph',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(build_graph_gesture_gesture_recognizer, 'graph_gesture_tiny')
