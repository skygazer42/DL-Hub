"""Instance segmentation models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_instance_segmenter(...)` factory and a `__main__` smoke test.
"""

from .blendmask import BlendMask, build_blendmask_instance_segmenter
from .boxinst import BoxInst, build_boxinst_instance_segmenter
from .cascade_mask_rcnn import CascadeMaskRCNN, build_cascade_mask_rcnn_instance_segmenter
from .centermask import CenterMask, build_centermask_instance_segmenter
from .condinst import CondInst, build_condinst_instance_segmenter
from .detr_mask import DETRMask, build_detr_mask_instance_segmenter
from .fcis import FCIS, build_fcis_instance_segmenter
from .htc import HTC, build_htc_instance_segmenter
from .mask2former import Mask2Former, build_mask2former_instance_segmenter
from .mask_rcnn import MaskRCNNInstanceSegmenter, build_mask_rcnn_instance_segmenter
from .mask_scoring_rcnn import MaskScoringRCNN, build_mask_scoring_rcnn_instance_segmenter
from .maskformer import MaskFormer, build_maskformer_instance_segmenter
from .point_rend import PointRendLite, build_pointrend_instance_segmenter
from .polarmask import PolarMask, build_polarmask_instance_segmenter
from .queryinst import QueryInst, build_queryinst_instance_segmenter
from .scnet import SCNet, build_scnet_instance_segmenter
from .solo import SOLO, build_solo_instance_segmenter
from .solov2 import SOLOv2, build_solov2_instance_segmenter
from .sparseinst import SparseInst, build_sparseinst_instance_segmenter
from .tensormask import TensorMask, build_tensormask_instance_segmenter
from .yolact import YOLACTLite, build_yolact_instance_segmenter

__all__ = [
    "BlendMask",
    "BoxInst",
    "CascadeMaskRCNN",
    "CenterMask",
    "CondInst",
    "DETRMask",
    "FCIS",
    "HTC",
    "Mask2Former",
    "MaskFormer",
    "MaskRCNNInstanceSegmenter",
    "MaskScoringRCNN",
    "PointRendLite",
    "PolarMask",
    "QueryInst",
    "SCNet",
    "SOLO",
    "SOLOv2",
    "SparseInst",
    "TensorMask",
    "YOLACTLite",
    "build_blendmask_instance_segmenter",
    "build_boxinst_instance_segmenter",
    "build_cascade_mask_rcnn_instance_segmenter",
    "build_centermask_instance_segmenter",
    "build_condinst_instance_segmenter",
    "build_detr_mask_instance_segmenter",
    "build_fcis_instance_segmenter",
    "build_htc_instance_segmenter",
    "build_mask2former_instance_segmenter",
    "build_mask_rcnn_instance_segmenter",
    "build_mask_scoring_rcnn_instance_segmenter",
    "build_maskformer_instance_segmenter",
    "build_pointrend_instance_segmenter",
    "build_polarmask_instance_segmenter",
    "build_queryinst_instance_segmenter",
    "build_scnet_instance_segmenter",
    "build_solo_instance_segmenter",
    "build_solov2_instance_segmenter",
    "build_sparseinst_instance_segmenter",
    "build_tensormask_instance_segmenter",
    "build_yolact_instance_segmenter",
]
