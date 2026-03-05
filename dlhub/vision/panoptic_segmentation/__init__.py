"""Panoptic segmentation models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_panoptic_segmenter(...)` factory and a `__main__` smoke test.
- Forward returns a dict with (at minimum) semantic logits and instance mask predictions.
"""

from .aunet import AUNet, build_aunet_panoptic_segmenter
from .axial_deeplab import AxialDeepLabPanoptic, build_axial_deeplab_panoptic_segmenter
from .bisenet_panoptic import BiSeNetPanoptic, build_bisenet_panoptic_segmenter
from .blendmask_panoptic import BlendMaskPanoptic, build_blendmask_panoptic_segmenter
from .boxinst_panoptic import BoxInstPanoptic, build_boxinst_panoptic_segmenter
from .centermask_panoptic import CenterMaskPanoptic, build_centermask_panoptic_segmenter
from .conditional_detr_panoptic import ConditionalDETRPanoptic, build_conditional_detr_panoptic_segmenter
from .condinst_panoptic import CondInstPanoptic, build_condinst_panoptic_segmenter
from .dab_detr_panoptic import DABDETRPanoptic, build_dab_detr_panoptic_segmenter
from .deformable_detr_panoptic import DeformableDETRPanoptic, build_deformable_detr_panoptic_segmenter
from .detr_panoptic import DETRPanoptic, build_detr_panoptic_segmenter
from .dn_detr_panoptic import DNDTRPanoptic, build_dn_detr_panoptic_segmenter
from .efficientps import EfficientPS, build_efficientps_panoptic_segmenter
from .hrnet_panoptic import HRNetPanoptic, build_hrnet_panoptic_segmenter
from .knet_panoptic import KNetPanoptic, build_knet_panoptic_segmenter
from .mask2former_panoptic import Mask2FormerPanoptic, build_mask2former_panoptic_segmenter
from .maskformer_panoptic import MaskFormerPanoptic, build_maskformer_panoptic_segmenter
from .max_deeplab_panoptic import MaXDeepLabPanoptic, build_max_deeplab_panoptic_segmenter
from .ocrnet_panoptic import OCRNetPanoptic, build_ocrnet_panoptic_segmenter
from .panoptic_deeplab import PanopticDeepLab, build_panoptic_deeplab_panoptic_segmenter
from .panoptic_fcn import PanopticFCN, build_panoptic_fcn_panoptic_segmenter
from .panoptic_fpn import PanopticFPN, build_panoptic_fpn_panoptic_segmenter
from .panoptic_segformer import PanopticSegFormer, build_panoptic_segformer_panoptic_segmenter
from .pointrend_panoptic import PointRendPanoptic, build_pointrend_panoptic_segmenter
from .polarmask_panoptic import PolarMaskPanoptic, build_polarmask_panoptic_segmenter
from .queryinst_panoptic import QueryInstPanoptic, build_queryinst_panoptic_segmenter
from .rt_detr_panoptic import RTDETRPanoptic, build_rtdetr_panoptic_segmenter
from .scnet_panoptic import SCNetPanoptic, build_scnet_panoptic_segmenter
from .setr_panoptic import SETRPanoptic, build_setr_panoptic_segmenter
from .solo_panoptic import SOLOPanoptic, build_solo_panoptic_segmenter
from .solov2_panoptic import SOLOv2Panoptic, build_solov2_panoptic_segmenter
from .sparse_rcnn_panoptic import SparseRCNNPanoptic, build_sparse_rcnn_panoptic_segmenter
from .sparseinst_panoptic import SparseInstPanoptic, build_sparseinst_panoptic_segmenter
from .tascnet import TASCNet, build_tascnet_panoptic_segmenter
from .tensormask_panoptic import TensorMaskPanoptic, build_tensormask_panoptic_segmenter
from .transunet_panoptic import TransUNetPanoptic, build_transunet_panoptic_segmenter
from .uberpanoptic import UberPanopticNet, build_uberpanoptic_panoptic_segmenter
from .upernet_panoptic import UPerNetPanoptic, build_upernet_panoptic_segmenter
from .upsnet import UPSNet, build_upsnet_panoptic_segmenter
from .yolact_panoptic import YOLACTPanoptic, build_yolact_panoptic_segmenter

__all__ = [
    "AUNet",
    "AxialDeepLabPanoptic",
    "BiSeNetPanoptic",
    "BlendMaskPanoptic",
    "BoxInstPanoptic",
    "CenterMaskPanoptic",
    "ConditionalDETRPanoptic",
    "CondInstPanoptic",
    "DABDETRPanoptic",
    "DeformableDETRPanoptic",
    "DETRPanoptic",
    "DNDTRPanoptic",
    "EfficientPS",
    "HRNetPanoptic",
    "KNetPanoptic",
    "Mask2FormerPanoptic",
    "MaskFormerPanoptic",
    "MaXDeepLabPanoptic",
    "OCRNetPanoptic",
    "PanopticDeepLab",
    "PanopticFCN",
    "PanopticFPN",
    "PanopticSegFormer",
    "PointRendPanoptic",
    "PolarMaskPanoptic",
    "QueryInstPanoptic",
    "RTDETRPanoptic",
    "SCNetPanoptic",
    "SETRPanoptic",
    "SOLOPanoptic",
    "SOLOv2Panoptic",
    "SparseRCNNPanoptic",
    "SparseInstPanoptic",
    "TASCNet",
    "TensorMaskPanoptic",
    "TransUNetPanoptic",
    "UPerNetPanoptic",
    "UberPanopticNet",
    "UPSNet",
    "YOLACTPanoptic",
    "build_aunet_panoptic_segmenter",
    "build_axial_deeplab_panoptic_segmenter",
    "build_bisenet_panoptic_segmenter",
    "build_blendmask_panoptic_segmenter",
    "build_boxinst_panoptic_segmenter",
    "build_centermask_panoptic_segmenter",
    "build_conditional_detr_panoptic_segmenter",
    "build_condinst_panoptic_segmenter",
    "build_dab_detr_panoptic_segmenter",
    "build_deformable_detr_panoptic_segmenter",
    "build_detr_panoptic_segmenter",
    "build_dn_detr_panoptic_segmenter",
    "build_efficientps_panoptic_segmenter",
    "build_hrnet_panoptic_segmenter",
    "build_knet_panoptic_segmenter",
    "build_mask2former_panoptic_segmenter",
    "build_maskformer_panoptic_segmenter",
    "build_max_deeplab_panoptic_segmenter",
    "build_ocrnet_panoptic_segmenter",
    "build_panoptic_deeplab_panoptic_segmenter",
    "build_panoptic_fcn_panoptic_segmenter",
    "build_panoptic_fpn_panoptic_segmenter",
    "build_panoptic_segformer_panoptic_segmenter",
    "build_pointrend_panoptic_segmenter",
    "build_polarmask_panoptic_segmenter",
    "build_queryinst_panoptic_segmenter",
    "build_rtdetr_panoptic_segmenter",
    "build_scnet_panoptic_segmenter",
    "build_setr_panoptic_segmenter",
    "build_solo_panoptic_segmenter",
    "build_solov2_panoptic_segmenter",
    "build_sparse_rcnn_panoptic_segmenter",
    "build_sparseinst_panoptic_segmenter",
    "build_tascnet_panoptic_segmenter",
    "build_tensormask_panoptic_segmenter",
    "build_transunet_panoptic_segmenter",
    "build_uberpanoptic_panoptic_segmenter",
    "build_upernet_panoptic_segmenter",
    "build_upsnet_panoptic_segmenter",
    "build_yolact_panoptic_segmenter",
]
