"""Instance segmentation models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_instance_segmenter(...)` factory and a `__main__` smoke test.
"""

from .bcnet import BCNet, build_bcnet_instance_segmenter
from .blendmask import BlendMask, build_blendmask_instance_segmenter
from .boxinst import BoxInst, build_boxinst_instance_segmenter
from .cascade_mask_rcnn import CascadeMaskRCNN, build_cascade_mask_rcnn_instance_segmenter
from .cfm import CFM, build_cfm_instance_segmenter
from .centermask import CenterMask, build_centermask_instance_segmenter
from .condinst import CondInst, build_condinst_instance_segmenter
from .dct_mask import DCTMask, build_dct_mask_instance_segmenter
from .deepmask import DeepMask, build_deepmask_instance_segmenter
from .deepsnake import DeepSnake, build_deepsnake_instance_segmenter
from .detr_mask import DETRMask, build_detr_mask_instance_segmenter
from .dynamicinst import DynamicInst, build_dynamicinst_instance_segmenter
from .e2ec import E2EC, build_e2ec_instance_segmenter
from .fastinst import FastInst, build_fastinst_instance_segmenter
from .fcis import FCIS, build_fcis_instance_segmenter
from .htc import HTC, build_htc_instance_segmenter
from .instancefcn import InstanceFCN, build_instancefcn_instance_segmenter
from .mask_dino import MaskDINO, build_mask_dino_instance_segmenter
from .mask2former import Mask2Former, build_mask2former_instance_segmenter
from .mask_rcnn import MaskRCNNInstanceSegmenter, build_mask_rcnn_instance_segmenter
from .mask_scoring_rcnn import MaskScoringRCNN, build_mask_scoring_rcnn_instance_segmenter
from .maskformer import MaskFormer, build_maskformer_instance_segmenter
from .meinst import MEInst, build_meinst_instance_segmenter
from .mnc import MNC, build_mnc_instance_segmenter
from .orienmask import OrienMask, build_orienmask_instance_segmenter
from .panet import PANet, build_panet_instance_segmenter
from .point_rend import PointRendLite, build_pointrend_instance_segmenter
from .polarmask import PolarMask, build_polarmask_instance_segmenter
from .queryinst import QueryInst, build_queryinst_instance_segmenter
from .refinemask import RefineMask, build_refinemask_instance_segmenter
from .rtmdet_ins import RTMDetIns, build_rtmdet_ins_instance_segmenter
from .scnet import SCNet, build_scnet_instance_segmenter
from .shapemask import ShapeMask, build_shapemask_instance_segmenter
from .sharpmask import SharpMask, build_sharpmask_instance_segmenter
from .sipmask import SipMask, build_sipmask_instance_segmenter
from .solo import SOLO, build_solo_instance_segmenter
from .solov2 import SOLOv2, build_solov2_instance_segmenter
from .sparseinst import SparseInst, build_sparseinst_instance_segmenter
from .tensormask import TensorMask, build_tensormask_instance_segmenter
from .yolact import YOLACTLite, build_yolact_instance_segmenter

__all__ = [
    "BCNet",
    "BlendMask",
    "BoxInst",
    "CascadeMaskRCNN",
    "CFM",
    "CenterMask",
    "CondInst",
    "DCTMask",
    "DeepMask",
    "DeepSnake",
    "DETRMask",
    "DynamicInst",
    "E2EC",
    "FastInst",
    "FCIS",
    "HTC",
    "InstanceFCN",
    "MaskDINO",
    "Mask2Former",
    "MaskFormer",
    "MaskRCNNInstanceSegmenter",
    "MaskScoringRCNN",
    "MEInst",
    "MNC",
    "OrienMask",
    "PANet",
    "PointRendLite",
    "PolarMask",
    "QueryInst",
    "RefineMask",
    "RTMDetIns",
    "SCNet",
    "ShapeMask",
    "SharpMask",
    "SipMask",
    "SOLO",
    "SOLOv2",
    "SparseInst",
    "TensorMask",
    "YOLACTLite",
    "build_bcnet_instance_segmenter",
    "build_blendmask_instance_segmenter",
    "build_boxinst_instance_segmenter",
    "build_cascade_mask_rcnn_instance_segmenter",
    "build_cfm_instance_segmenter",
    "build_centermask_instance_segmenter",
    "build_condinst_instance_segmenter",
    "build_dct_mask_instance_segmenter",
    "build_deepmask_instance_segmenter",
    "build_deepsnake_instance_segmenter",
    "build_detr_mask_instance_segmenter",
    "build_dynamicinst_instance_segmenter",
    "build_e2ec_instance_segmenter",
    "build_fastinst_instance_segmenter",
    "build_fcis_instance_segmenter",
    "build_htc_instance_segmenter",
    "build_instancefcn_instance_segmenter",
    "build_mask_dino_instance_segmenter",
    "build_mask2former_instance_segmenter",
    "build_mask_rcnn_instance_segmenter",
    "build_mask_scoring_rcnn_instance_segmenter",
    "build_maskformer_instance_segmenter",
    "build_meinst_instance_segmenter",
    "build_mnc_instance_segmenter",
    "build_orienmask_instance_segmenter",
    "build_panet_instance_segmenter",
    "build_pointrend_instance_segmenter",
    "build_polarmask_instance_segmenter",
    "build_queryinst_instance_segmenter",
    "build_refinemask_instance_segmenter",
    "build_rtmdet_ins_instance_segmenter",
    "build_scnet_instance_segmenter",
    "build_shapemask_instance_segmenter",
    "build_sharpmask_instance_segmenter",
    "build_sipmask_instance_segmenter",
    "build_solo_instance_segmenter",
    "build_solov2_instance_segmenter",
    "build_sparseinst_instance_segmenter",
    "build_tensormask_instance_segmenter",
    "build_yolact_instance_segmenter",
]
