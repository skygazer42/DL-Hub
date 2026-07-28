"""Semantic segmentation models (pure torch, compact-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_segmenter(...)` factory and a `__main__` smoke test.
"""

from .bisenetv1 import BiSeNetV1, build_bisenetv1_segmenter
from .bisenetv2 import BiSeNetV2, build_bisenetv2_segmenter
from .cgnet import CGNet, build_cgnet_segmenter
from .danet import DANet, build_danet_segmenter
from .deeplabv3 import DeepLabV3, build_deeplabv3_segmenter
from .deeplabv3plus import DeepLabV3Plus, build_deeplabv3plus_segmenter
from .enet import ENet, build_enet_segmenter
from .erfnet import ERFNet, build_erfnet_segmenter
from .espnet import ESPNet, build_espnet_segmenter
from .espnetv2 import ESPNetV2, build_espnetv2_segmenter
from .fastscnn import FastSCNN, build_fastscnn_segmenter
from .fcn import FCNSegmenter, build_fcn_segmenter
from .icnet import ICNet, build_icnet_segmenter
from .lednet import LEDNet, build_lednet_segmenter
from .linknet import LinkNet, build_linknet_segmenter
from .ocrnet import OCRNet, build_ocrnet_segmenter
from .pspnet import PSPNet, build_pspnet_segmenter
from .refinenet import RefineNet, build_refinenet_segmenter
from .segformer import SegFormer, build_segformer_segmenter
from .segnet import SegNet, build_segnet_segmenter
from .transunet import TransUNet, build_transunet_segmenter
from .unet import UNetSegmenter, build_unet_segmenter
from .upernet import UPerNet, build_upernet_segmenter

__all__ = [
    "BiSeNetV1",
    "BiSeNetV2",
    "CGNet",
    "DANet",
    "DeepLabV3",
    "DeepLabV3Plus",
    "ENet",
    "ERFNet",
    "ESPNet",
    "ESPNetV2",
    "FCNSegmenter",
    "FastSCNN",
    "ICNet",
    "LEDNet",
    "LinkNet",
    "OCRNet",
    "PSPNet",
    "RefineNet",
    "SegFormer",
    "SegNet",
    "TransUNet",
    "UNetSegmenter",
    "UPerNet",
    "build_bisenetv1_segmenter",
    "build_bisenetv2_segmenter",
    "build_cgnet_segmenter",
    "build_danet_segmenter",
    "build_deeplabv3_segmenter",
    "build_deeplabv3plus_segmenter",
    "build_enet_segmenter",
    "build_erfnet_segmenter",
    "build_espnet_segmenter",
    "build_espnetv2_segmenter",
    "build_fastscnn_segmenter",
    "build_fcn_segmenter",
    "build_icnet_segmenter",
    "build_lednet_segmenter",
    "build_linknet_segmenter",
    "build_ocrnet_segmenter",
    "build_pspnet_segmenter",
    "build_refinenet_segmenter",
    "build_segformer_segmenter",
    "build_segnet_segmenter",
    "build_transunet_segmenter",
    "build_unet_segmenter",
    "build_upernet_segmenter",
]
