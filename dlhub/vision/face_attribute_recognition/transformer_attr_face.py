from __future__ import annotations

from torch import nn

from ._common import build_toy_face_attribute_recognizer, smoke_test_face_attribute_recognizer


_VARIANTS: dict[str, dict[str, int]] = {'transformer_attr_face_tiny': {'width': 24, 'depth': 1}, 'transformer_attr_face_small': {'width': 36, 'depth': 2}, 'transformer_attr_face_base': {'width': 48, 'depth': 3}}


def build_transformer_attr_face_attribute_recognizer(
    *,
    in_channels: int,
    variant: str = 'transformer_attr_face_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_attribute_recognizer(
        family='transformer_attr_face',
        mode='transformer',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_attribute_recognizer(build_transformer_attr_face_attribute_recognizer, 'transformer_attr_face_tiny')
