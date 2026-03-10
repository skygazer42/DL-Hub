"""Object detection timeline metadata (best-effort, for docs/CLI).

Notes:
- Years are based on representative papers or the earliest commonly-cited version.
- Models in this repo are toy-first interpretations of detector families, not exact reproductions.
- The timeline spans the modern archive from 2014 onward; if a family has no stable metadata,
  prefer updating this file rather than hardcoding logic elsewhere.
"""

from dataclasses import dataclass

ARCHIVE_START_YEAR = 2014
ARCHIVE_END_YEAR = 2026


@dataclass(frozen=True)
class TimelineEntry:
    year: int | None
    family: str
    method: str
    group: str
    reference: str | None = None


_ENTRIES: list[TimelineEntry] = [
    TimelineEntry(2014, "overfeat", "OverFeat (multi-scale conv detector, toy)", "single_stage"),
    TimelineEntry(2014, "rcnn", "R-CNN (region proposals + CNN classifier, toy)", "two_stage"),
    TimelineEntry(2014, "sppnet", "SPPNet (spatial pyramid pooling detector, toy)", "two_stage"),
    TimelineEntry(2015, "faster_rcnn", "Faster R-CNN (RPN + RoI head, toy)", "two_stage"),
    TimelineEntry(2015, "fast_rcnn", "Fast R-CNN (shared conv feature detector, toy)", "two_stage"),
    TimelineEntry(
        2015, "densebox", "DenseBox (dense regression detector, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(2016, "ssd", "SSD (single-shot multi-scale detector)", "single_stage"),
    TimelineEntry(
        2016, "squeezedet", "SqueezeDet (compact single-shot detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2016, "rfcn", "R-FCN (region-based fully convolutional detector, toy)", "two_stage"
    ),
    TimelineEntry(
        2016, "unitbox", "UnitBox (IoU-aware box regression, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(2016, "hypernet", "HyperNet (hyper-feature region detector, toy)", "two_stage"),
    TimelineEntry(2016, "yolo", "YOLOv1 (single-stage grid detector, toy)", "single_stage"),
    TimelineEntry(2017, "dssd", "DSSD (deconvolutional SSD, toy)", "single_stage"),
    TimelineEntry(2017, "retinanet", "RetinaNet (focal loss one-stage detector)", "single_stage"),
    TimelineEntry(
        2017, "mask_rcnn", "Mask R-CNN (instance-aware two-stage detector, toy)", "two_stage"
    ),
    TimelineEntry(2017, "yolov2", "YOLOv2 (anchor-based YOLO, toy)", "single_stage"),
    TimelineEntry(
        2017, "ron", "RON (reverse connection with objectness prior, toy)", "single_stage"
    ),
    TimelineEntry(
        2017,
        "point_linking_network",
        "Point Linking Network (point-based detector, toy)",
        "keypoint_anchor_free",
    ),
    TimelineEntry(
        2018, "cornernet", "CornerNet (paired keypoint detector, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(2018, "cascade_rcnn", "Cascade R-CNN (multi-stage refinement, toy)", "two_stage"),
    TimelineEntry(2018, "yolov3", "YOLOv3 (multi-scale YOLO, toy)", "single_stage"),
    TimelineEntry(
        2018, "refinedet", "RefineDet (refinement-based one-stage detector, toy)", "single_stage"
    ),
    TimelineEntry(2019, "atss", "ATSS (adaptive sample selection)", "single_stage"),
    TimelineEntry(2019, "centernet", "CenterNet (objects as points, toy)", "keypoint_anchor_free"),
    TimelineEntry(
        2019, "extremenet", "ExtremeNet (extreme-point detector, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(
        2019, "fcos", "FCOS (fully convolutional one-stage detector, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(
        2019, "foveabox", "FoveaBox (foveated anchor-free detector, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(
        2019, "freeanchor", "FreeAnchor (anchor assignment refinement, toy)", "single_stage"
    ),
    TimelineEntry(2019, "reppoints", "RepPoints (point-set detector, toy)", "keypoint_anchor_free"),
    TimelineEntry(
        2019, "tridentnet", "TridentNet (scale-aware multi-branch detector, toy)", "two_stage"
    ),
    TimelineEntry(
        2019, "libra_rcnn", "Libra R-CNN (balanced sampling and heads, toy)", "two_stage"
    ),
    TimelineEntry(2019, "grid_rcnn", "Grid R-CNN (grid-guided box localization, toy)", "two_stage"),
    TimelineEntry(
        2019,
        "guided_anchoring_rcnn",
        "Guided Anchoring R-CNN (dynamic anchor guidance, toy)",
        "two_stage",
    ),
    TimelineEntry(
        2019, "detectors", "DetectoRS (recursive feature pyramid detector, toy)", "two_stage"
    ),
    TimelineEntry(
        2019, "m2det", "M2Det (multi-level feature pyramid detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2020, "detr", "DETR (set prediction with transformer queries)", "transformer_query"
    ),
    TimelineEntry(
        2020, "efficientdet", "EfficientDet (BiFPN one-stage detector, toy)", "single_stage"
    ),
    TimelineEntry(2020, "gfl", "GFL (generalized focal loss detector, toy)", "single_stage"),
    TimelineEntry(2020, "nanodet", "NanoDet (lightweight GFL-style detector, toy)", "single_stage"),
    TimelineEntry(
        2020, "paa", "PAA (probabilistic anchor assignment detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2020, "ttfnet", "TTFNet (keypoint heatmap detector, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(2020, "yolov5", "YOLOv5 (CSP + PAN detector, toy)", "single_stage"),
    TimelineEntry(2020, "yolov4", "YOLOv4 (CSPDarknet detector, toy)", "single_stage"),
    TimelineEntry(
        2020, "ppyolo", "PP-YOLO (bag-of-freebies single-stage detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2020,
        "borderdet",
        "BorderDet (border-aware anchor-free detector, toy)",
        "keypoint_anchor_free",
    ),
    TimelineEntry(
        2020,
        "autoassign",
        "AutoAssign (fully differentiable label assignment, toy)",
        "keypoint_anchor_free",
    ),
    TimelineEntry(
        2020, "dynamic_rcnn", "Dynamic R-CNN (adaptive training detector, toy)", "two_stage"
    ),
    TimelineEntry(
        2021,
        "conditional_detr",
        "Conditional DETR (conditional query detector, toy)",
        "transformer_query",
    ),
    TimelineEntry(
        2021,
        "deformable_detr",
        "Deformable DETR (multi-scale deformable queries, toy)",
        "transformer_query",
    ),
    TimelineEntry(2021, "dn_detr", "DN-DETR (denoising query detector, toy)", "transformer_query"),
    TimelineEntry(
        2021, "fsaf", "FSAF (feature selective anchor-free detector, toy)", "keypoint_anchor_free"
    ),
    TimelineEntry(2021, "tood", "TOOD (task-aligned one-stage detector, toy)", "single_stage"),
    TimelineEntry(
        2021, "varifocalnet", "VarifocalNet (quality-aware one-stage detector, toy)", "single_stage"
    ),
    TimelineEntry(2021, "vfnet", "VFNet (varifocal dense detector, toy)", "single_stage"),
    TimelineEntry(2021, "yolof", "YOLOF (single-level one-stage detector, toy)", "single_stage"),
    TimelineEntry(
        2021, "yolox", "YOLOX (anchor-free decoupled head detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2021, "smca_detr", "SMCA-DETR (spatially modulated co-attention, toy)", "transformer_query"
    ),
    TimelineEntry(
        2021, "yolos", "YOLOS (ViT-style detector with tokens, toy)", "transformer_query"
    ),
    TimelineEntry(
        2021, "scaled_yolov4", "Scaled-YOLOv4 (scaled YOLO detector, toy)", "single_stage"
    ),
    TimelineEntry(2021, "ppyolov2", "PP-YOLOv2 (improved PP-YOLO detector, toy)", "single_stage"),
    TimelineEntry(
        2021, "giraffedet", "GiraffeDet (lightweight feature fusion detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2021,
        "centernet2",
        "CenterNet2 (proposal-enhanced point detector, toy)",
        "keypoint_anchor_free",
    ),
    TimelineEntry(
        2021,
        "vild",
        "ViLD (open-vocabulary distillation detector, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2022, "dab_detr", "DAB-DETR (dynamic anchor boxes as queries, toy)", "transformer_query"
    ),
    TimelineEntry(
        2022,
        "dino",
        "DINO (denoising anchor boxes with contrastive learning, toy)",
        "transformer_query",
    ),
    TimelineEntry(2022, "ppyoloe", "PP-YOLOE (anchor-free PP-YOLO family, toy)", "single_stage"),
    TimelineEntry(2022, "rtmdet", "RTMDet (real-time dense detector, toy)", "single_stage"),
    TimelineEntry(
        2022, "sparse_rcnn", "Sparse R-CNN (learned proposal queries)", "transformer_query"
    ),
    TimelineEntry(
        2022, "anchor_detr", "Anchor DETR (anchor-guided queries, toy)", "transformer_query"
    ),
    TimelineEntry(
        2022, "adamixer", "AdaMixer (adaptive query mixing detector, toy)", "transformer_query"
    ),
    TimelineEntry(
        2022,
        "efficient_detr",
        "Efficient DETR (efficient transformer detector, toy)",
        "transformer_query",
    ),
    TimelineEntry(2022, "deta", "DETA (deformable transformer detector, toy)", "transformer_query"),
    TimelineEntry(2022, "yolov6", "YOLOv6 (industrial real-time detector, toy)", "single_stage"),
    TimelineEntry(2022, "yolov7", "YOLOv7 (E-ELAN real-time detector, toy)", "single_stage"),
    TimelineEntry(
        2022, "damo_yolo", "DAMO-YOLO (NAS/lightweight YOLO detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2022,
        "regionclip",
        "RegionCLIP (region-text aligned detector, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2022,
        "glip",
        "GLIP (grounded language-image pretraining detector, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2022,
        "detclip",
        "DetCLIP (open-vocabulary contrastive detector, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2023, "rtdetr", "RT-DETR (real-time transformer detector, toy)", "transformer_query"
    ),
    TimelineEntry(2023, "yolov8", "YOLOv8 (C2f anchor-free detector, toy)", "single_stage"),
    TimelineEntry(
        2023,
        "co_detr",
        "Co-DETR (collaborative hybrid assignment detector, toy)",
        "transformer_query",
    ),
    TimelineEntry(
        2023,
        "group_detr",
        "Group DETR (grouped query optimization detector, toy)",
        "transformer_query",
    ),
    TimelineEntry(
        2023, "h_detr", "H-DETR (hybrid matching DETR variant, toy)", "transformer_query"
    ),
    TimelineEntry(
        2023, "gold_yolo", "Gold-YOLO (efficient aggregation detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2023,
        "owl_vit",
        "OWL-ViT (open-vocabulary vision transformer detector, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2023,
        "grounding_dino",
        "Grounding DINO (grounded open-set detector, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2023,
        "detclipv2",
        "DetCLIPv2 (scalable open-vocabulary detector, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2024, "yolov9", "YOLOv9 (programmable gradient YOLO detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2024, "yolov10", "YOLOv10 (NMS-free real-time YOLO detector, toy)", "single_stage"
    ),
    TimelineEntry(
        2024, "yolo11", "YOLO11 (Ultralytics unified detector family, toy)", "single_stage"
    ),
    TimelineEntry(
        2024,
        "d_fine",
        "D-FINE (real-time detector with distribution refinement, toy)",
        "transformer_query",
    ),
    TimelineEntry(
        2024, "lw_detr", "LW-DETR (lightweight DETR-style detector, toy)", "transformer_query"
    ),
    TimelineEntry(
        2024,
        "ovlw_detr",
        "OVLW-DETR (open-vocabulary lightweight DETR, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(2024, "ddq_detr", "DDQ-DETR (dense dynamic queries, toy)", "transformer_query"),
    TimelineEntry(
        2024,
        "yolo_world",
        "YOLO-World (real-time open-vocabulary YOLO, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2024,
        "decola",
        "DECOLA (language-conditioned detection transformer, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(2025, "yolo12", "YOLO12 (attention-centric YOLO detector, toy)", "single_stage"),
    TimelineEntry(2025, "yolo13", "YOLO13 (next-generation YOLO detector, toy)", "single_stage"),
    TimelineEntry(
        2025,
        "rtgen",
        "RTGen (real-time open-ended detection with generative VLMs, toy)",
        "open_vocabulary_multimodal",
    ),
    TimelineEntry(
        2025, "sa_detr", "SA-DETR (scale-adaptive DETR-style detector, toy)", "transformer_query"
    ),
    TimelineEntry(
        2026, "yolo26", "YOLO26 (latest Ultralytics real-time detector, toy)", "single_stage"
    ),
]

_CANONICAL_FAMILY_ALIASES: dict[str, str] = {
    "fcos_nocenter": "fcos",
    "yolo_v1": "yolo",
}

_EXAMPLE_ARCH_IDS: dict[str, str] = {
    "fcos": "dldet:fcos_tiny",
    "yolo": "dldet:yolo_v1_tiny",
}

_FAMILY_SERIES: dict[str, str] = {
    "adamixer": "DETR-like / 查询集",
    "anchor_detr": "DETR-like / 查询集",
    "atss": "Dense one-stage / 密集单阶段",
    "autoassign": "Anchor-free point / 锚框自由",
    "borderdet": "Anchor-free point / 锚框自由",
    "cascade_rcnn": "R-CNN lineage / 双阶段",
    "centernet": "Center-based / 中心点",
    "centernet2": "Center-based / 中心点",
    "co_detr": "DETR-like / 查询集",
    "conditional_detr": "DETR-like / 查询集",
    "cornernet": "Keypoint pair / 关键点",
    "dab_detr": "DETR-like / 查询集",
    "damo_yolo": "YOLO-like / 单阶段",
    "ddq_detr": "DETR-like / 查询集",
    "decola": "Vision-language / 开放词汇",
    "deformable_detr": "DETR-like / 查询集",
    "d_fine": "DETR-like / 查询集",
    "densebox": "Anchor-free point / 锚框自由",
    "deta": "DETR-like / 查询集",
    "detclip": "Vision-language / 开放词汇",
    "detclipv2": "Vision-language / 开放词汇",
    "detectors": "R-CNN lineage / 双阶段",
    "detr": "DETR-like / 查询集",
    "dino": "DETR-like / 查询集",
    "dn_detr": "DETR-like / 查询集",
    "dssd": "SSD-like / 单阶段",
    "dynamic_rcnn": "R-CNN lineage / 双阶段",
    "efficient_detr": "DETR-like / 查询集",
    "efficientdet": "Dense one-stage / 密集单阶段",
    "extremenet": "Keypoint pair / 关键点",
    "fast_rcnn": "R-CNN lineage / 双阶段",
    "faster_rcnn": "R-CNN lineage / 双阶段",
    "fcos": "Anchor-free dense / 锚框自由",
    "foveabox": "Anchor-free dense / 锚框自由",
    "freeanchor": "Dense one-stage / 密集单阶段",
    "fsaf": "Anchor-free dense / 锚框自由",
    "gfl": "Dense one-stage / 密集单阶段",
    "giraffedet": "Dense one-stage / 密集单阶段",
    "glip": "Vision-language / 开放词汇",
    "gold_yolo": "YOLO-like / 单阶段",
    "grid_rcnn": "R-CNN lineage / 双阶段",
    "grounding_dino": "Vision-language / 开放词汇",
    "group_detr": "DETR-like / 查询集",
    "guided_anchoring_rcnn": "R-CNN lineage / 双阶段",
    "h_detr": "DETR-like / 查询集",
    "hypernet": "R-CNN lineage / 双阶段",
    "libra_rcnn": "R-CNN lineage / 双阶段",
    "lw_detr": "DETR-like / 查询集",
    "m2det": "Dense one-stage / 密集单阶段",
    "mask_rcnn": "R-CNN lineage / 双阶段",
    "nanodet": "Dense one-stage / 密集单阶段",
    "overfeat": "Dense one-stage / 密集单阶段",
    "owl_vit": "Vision-language / 开放词汇",
    "ovlw_detr": "Vision-language / 开放词汇",
    "paa": "Dense one-stage / 密集单阶段",
    "point_linking_network": "Anchor-free point / 锚框自由",
    "ppyolo": "YOLO-like / 单阶段",
    "ppyoloe": "YOLO-like / 单阶段",
    "ppyolov2": "YOLO-like / 单阶段",
    "rcnn": "R-CNN lineage / 双阶段",
    "refinedet": "SSD-like / 单阶段",
    "regionclip": "Vision-language / 开放词汇",
    "reppoints": "Anchor-free point / 锚框自由",
    "retinanet": "Dense one-stage / 密集单阶段",
    "rfcn": "R-CNN lineage / 双阶段",
    "ron": "Dense one-stage / 密集单阶段",
    "rtdetr": "DETR-like / 查询集",
    "rtmdet": "Dense one-stage / 密集单阶段",
    "rtgen": "Vision-language / 开放词汇",
    "sa_detr": "DETR-like / 查询集",
    "scaled_yolov4": "YOLO-like / 单阶段",
    "smca_detr": "DETR-like / 查询集",
    "sparse_rcnn": "R-CNN lineage / 双阶段",
    "sppnet": "R-CNN lineage / 双阶段",
    "squeezedet": "Dense one-stage / 密集单阶段",
    "ssd": "SSD-like / 单阶段",
    "tood": "Dense one-stage / 密集单阶段",
    "tridentnet": "R-CNN lineage / 双阶段",
    "ttfnet": "Center-based / 中心点",
    "unitbox": "Anchor-free dense / 锚框自由",
    "varifocalnet": "Dense one-stage / 密集单阶段",
    "vfnet": "Dense one-stage / 密集单阶段",
    "vild": "Vision-language / 开放词汇",
    "yolo": "YOLO-like / 单阶段",
    "yolo11": "YOLO-like / 单阶段",
    "yolo12": "YOLO-like / 单阶段",
    "yolo13": "YOLO-like / 单阶段",
    "yolo26": "YOLO-like / 单阶段",
    "yolo_world": "Vision-language / 开放词汇",
    "yolof": "YOLO-like / 单阶段",
    "yolos": "DETR-like / 查询集",
    "yolov10": "YOLO-like / 单阶段",
    "yolov2": "YOLO-like / 单阶段",
    "yolov3": "YOLO-like / 单阶段",
    "yolov4": "YOLO-like / 单阶段",
    "yolov5": "YOLO-like / 单阶段",
    "yolov6": "YOLO-like / 单阶段",
    "yolov7": "YOLO-like / 单阶段",
    "yolov8": "YOLO-like / 单阶段",
    "yolov9": "YOLO-like / 单阶段",
    "yolox": "YOLO-like / 单阶段",
}


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}


def canonical_family_name(name: str) -> str:
    family = str(name).strip().lower()
    for suffix in ("_tiny", "_small", "_base"):
        if family.endswith(suffix):
            family = family[: -len(suffix)]
            break
    return _CANONICAL_FAMILY_ALIASES.get(family, family)


def example_arch_id(family: str) -> str:
    canonical = canonical_family_name(family)
    return _EXAMPLE_ARCH_IDS.get(canonical, f"dldet:{canonical}_tiny")


def family_series_label(family: str) -> str:
    canonical = canonical_family_name(family)
    return _FAMILY_SERIES.get(canonical, "General detector / 通用检测")
