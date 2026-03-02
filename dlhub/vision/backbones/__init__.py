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
from .extra_cnn import (
    build_alexnet_classifier,
    build_cspdarknet_classifier,
    build_darknet_classifier,
    build_ghostnet_classifier,
    build_googlenet_classifier,
    build_lenet_classifier,
    build_mnasnet_classifier,
    build_mobileone_classifier,
    build_nin_classifier,
    build_regnet_classifier,
    build_shufflenet_v1_classifier,
    build_xception_classifier,
    build_zfnet_classifier,
)
from .mixers import build_gmlp_classifier, build_poolformer_classifier, build_resmlp_classifier
from .swin import build_swin_classifier
from .transformers import build_convmixer_classifier, build_mlp_mixer_classifier, build_vit_classifier
from .hybrids import build_coatnet_classifier, build_fnet_classifier, build_mobilevit_classifier

__all__ = [
    "build_alexnet_classifier",
    "build_coatnet_classifier",
    "build_convmixer_classifier",
    "build_convnext_classifier",
    "build_cspdarknet_classifier",
    "build_darknet_classifier",
    "build_densenet_classifier",
    "build_efficientnet_classifier",
    "build_fnet_classifier",
    "build_ghostnet_classifier",
    "build_googlenet_classifier",
    "build_lenet_classifier",
    "build_gmlp_classifier",
    "build_mlp_mixer_classifier",
    "build_mobilenet_v1_classifier",
    "build_mobilenet_v2_classifier",
    "build_mobilenet_v3_classifier",
    "build_mobilevit_classifier",
    "build_mnasnet_classifier",
    "build_mobileone_classifier",
    "build_nin_classifier",
    "build_poolformer_classifier",
    "build_regnet_classifier",
    "build_repvgg_classifier",
    "build_resnet_classifier",
    "build_resmlp_classifier",
    "build_shufflenet_v1_classifier",
    "build_shufflenet_v2_classifier",
    "build_squeezenet_classifier",
    "build_swin_classifier",
    "build_vgg_classifier",
    "build_vit_classifier",
    "build_xception_classifier",
    "build_zfnet_classifier",
]
