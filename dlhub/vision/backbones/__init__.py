"""Backbone architectures implemented in this repo (no downloads, no pretrained weights).

These backbones are intentionally lightweight and readable, so they can be reused across
multiple tracks/lessons without duplicating model code.
"""

from .cnn import (
    build_convnext_classifier,
    build_densenet_classifier,
    build_efficientnet_classifier,
    build_mobilenet_v1_classifier,
    build_mobilenet_v2_classifier,
    build_mobilenet_v3_classifier,
    build_repvgg_classifier,
    build_resnet_classifier,
    build_shufflenet_v2_classifier,
    build_squeezenet_classifier,
    build_vgg_classifier,
)
from .transformers import build_convmixer_classifier, build_mlp_mixer_classifier, build_vit_classifier

__all__ = [
    "build_convmixer_classifier",
    "build_convnext_classifier",
    "build_densenet_classifier",
    "build_efficientnet_classifier",
    "build_mlp_mixer_classifier",
    "build_mobilenet_v1_classifier",
    "build_mobilenet_v2_classifier",
    "build_mobilenet_v3_classifier",
    "build_repvgg_classifier",
    "build_resnet_classifier",
    "build_shufflenet_v2_classifier",
    "build_squeezenet_classifier",
    "build_vgg_classifier",
    "build_vit_classifier",
]

