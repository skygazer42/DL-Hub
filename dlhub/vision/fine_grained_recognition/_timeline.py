"""FGVC timeline metadata (best-effort, for docs/CLI).

Notes:
- Years are based on representative papers or the earliest commonly-cited version.
- Some families in this repo are "toy interpretations" of an idea, not strict reproductions.
- If you spot a mismatch, please open an issue/PR with the corrected reference.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEntry:
    year: int | None
    family: str
    method: str
    group: str
    reference: str | None = None


# Source of truth for `scripts/fine_grained_recognition_zoo.py --timeline`
# and the README "archive" section.
_ENTRIES: list[TimelineEntry] = [
    TimelineEntry(2014, "part_rcnn", "Part-based R-CNN (parts + crop)", "part"),
    TimelineEntry(2015, "bilinear_cnn", "Bilinear CNN (B-CNN)", "bilinear"),
    TimelineEntry(2016, "compact_bilinear", "Compact Bilinear Pooling", "bilinear"),
    TimelineEntry(2016, "part_stacked_cnn", "Part-Stacked CNN", "part"),
    TimelineEntry(2017, "kernel_pooling", "Kernel Pooling (bilinear variant)", "bilinear"),
    TimelineEntry(2017, "lowrank_bilinear", "Low-rank / Factorized Bilinear Pooling", "bilinear"),
    TimelineEntry(2017, "pa_cnn", "PA-CNN (part-aligned / part-attention CNN)", "part"),
    TimelineEntry(2017, "racnn", "RA-CNN (recurrent attention)", "part"),
    TimelineEntry(2017, "ma_cnn", "MA-CNN (multi-attention)", "part"),
    TimelineEntry(2017, "ga_cnn", "GA-CNN (granularity-aware)", "relation"),
    TimelineEntry(2017, "interp_parts", "Interpretable Part Modeling", "relation"),
    TimelineEntry(2018, "hierarchical_bilinear", "Hierarchical Bilinear Pooling", "bilinear"),
    TimelineEntry(
        2018, "isqrt_cov", "iSQRT-COV (iterative matrix sqrt for covariance pooling)", "bilinear"
    ),
    TimelineEntry(2018, "mpn_cov", "MPN-COV (matrix power normalized covariance)", "bilinear"),
    TimelineEntry(2018, "ws_ban", "WS-BAN (weakly supervised bilinear attention)", "bilinear"),
    TimelineEntry(2018, "osme_mamc", "OSME + MAMC (multi-attention + constraint)", "relation"),
    TimelineEntry(2018, "hse", "HSE (hierarchical semantic embedding)", "relation"),
    TimelineEntry(2018, "nts_net", "NTS-Net (proposal + zoom-in)", "part"),
    TimelineEntry(2018, "gem_pooling", "GeM pooling head (strong baseline)", "bilinear"),
    TimelineEntry(2019, "dfl_cnn", "DFL-CNN (discriminative filter learning)", "part"),
    TimelineEntry(2019, "tasn", "TASN (trilinear attention sampling)", "part"),
    TimelineEntry(2019, "s3n", "S3N (snapshot + zoom)", "part"),
    TimelineEntry(2019, "mge_cnn", "MGE-CNN (multi-granularity ensemble)", "part"),
    TimelineEntry(2019, "dcl", "DCL (deep complementary learning)", "relation"),
    TimelineEntry(2019, "ws_dan", "WS-DAN (weakly supervised data augmentation)", "relation"),
    TimelineEntry(2019, "proto_pnet", "ProtoPNet (prototype-based interpretability)", "relation"),
    TimelineEntry(2019, "partnet", "PartNet (part mining / part discovery)", "part"),
    TimelineEntry(2019, "api_net", "API-Net (attentive pairwise interaction)", "relation"),
    TimelineEntry(2019, "crossx", "CrossX (cross-region interaction)", "relation"),
    TimelineEntry(2020, "pmg", "PMG (progressive multi-granularity)", "part"),
    TimelineEntry(2020, "region_grouping", "Region Grouping (grouped regions/parts)", "relation"),
    TimelineEntry(2020, "pca_net", "PCA-Net (co-attention style transformer)", "transformer"),
    TimelineEntry(2021, "transfg", "TransFG (token selection for FGVC)", "transformer"),
    TimelineEntry(2021, "ffvt", "FFVT (fine-grained feature fusion ViT)", "transformer"),
    TimelineEntry(2021, "sim_trans", "Sim-Trans (similarity-driven transformer)", "transformer"),
    TimelineEntry(2021, "pim", "PIM (plug-in / part interaction module)", "transformer"),
    TimelineEntry(2021, "cvl", "CVL (vision-language token fusion)", "transformer"),
    TimelineEntry(2021, "pedtrans", "PedTrans (pose/metadata guided transformer)", "transformer"),
    TimelineEntry(
        2022, "metaformer_fgvc", "MetaFormer (backbone-style transformer family)", "transformer"
    ),
    TimelineEntry(2022, "vit_fod", "ViT-FOD (feature/object difference token)", "transformer"),
    TimelineEntry(2022, "aftrans", "AFTrans (attention fusion transformer)", "transformer"),
    TimelineEntry(2022, "vpt", "VPT (Visual Prompt Tuning)", "transformer"),
    TimelineEntry(2023, "sm_vit", "SM-ViT (salient mask guided ViT)", "transformer"),
    TimelineEntry(2024, "ldh_vit", "LDH-ViT (local concealment + selection)", "transformer"),
    TimelineEntry(
        2025, "prompt_cam", "Prompt-CAM (interpretable prompt attention map)", "transformer"
    ),
    TimelineEntry(2025, "fg_clip", "FG-CLIP (CLIP-style visual-text alignment)", "transformer"),
    TimelineEntry(
        2025, "finer_cam", "Finer-CAM (difference spotting for explanation)", "transformer"
    ),
    TimelineEntry(
        2025, "xr_vlm", "XR-VLM (multi-part prompts + cross-relationship modeling)", "transformer"
    ),
    TimelineEntry(2025, "gft", "GFT (graph-guided fine-tuning transformer)", "transformer"),
    TimelineEntry(
        2025,
        "e_finer",
        "E-FineR (efficient vocabulary-free FGVC with enriched grounding)",
        "transformer",
    ),
    TimelineEntry(
        2025, "unifgvc", "UniFGVC (universal fine-grained category discovery)", "transformer"
    ),
    TimelineEntry(
        2025,
        "granvit",
        "GranViT (hierarchical granularity-aware vision transformer)",
        "transformer",
    ),
    TimelineEntry(2025, "saccadic_vision", "Saccadic Vision (glimpse-based adaptive FGVC)", "part"),
    TimelineEntry(
        2025, "causal_fsfg", "CausalFSFG (causal few-shot fine-grained recognition)", "relation"
    ),
    TimelineEntry(
        2025, "micro_clip", "MicroCLIP (small CLIP-style FGVC adaptation)", "transformer"
    ),
    TimelineEntry(2025, "dcnn_fg", "DCNN-FG (dual-cross current network for FGVC)", "relation"),
    TimelineEntry(
        2025,
        "hfcr_net",
        "HFCR-Net (hierarchical feature calibration and relation learning)",
        "relation",
    ),
    TimelineEntry(
        2025,
        "cmcp_meta",
        "CMCP-Meta (connecting meta via cross-contrastive pretraining)",
        "relation",
    ),
    TimelineEntry(
        2025, "ficnet", "FICNet (fine-grained instance-centric compositional network)", "part"
    ),
    TimelineEntry(2025, "gcpl", "GCPL (generative class prompt learning for FGVC)", "transformer"),
    TimelineEntry(2025, "comple", "CoMPLe (cross-modal prompt learning for FGVC)", "transformer"),
    TimelineEntry(
        2025, "pp_ssl", "PP-SSL (part-prototype self-supervised learning for few-shot FGVC)", "part"
    ),
    TimelineEntry(
        2025,
        "part_rel_transformer",
        "PART (part-guided relational transformer for FGVC)",
        "relation",
    ),
    TimelineEntry(
        2025,
        "highorder_graph",
        "HighOrderGraph (graph-based high-order relation discovery for FGVC)",
        "relation",
    ),
    TimelineEntry(
        2025, "part_matching", "Part Matching (few-shot FGVC with explicit part matching)", "part"
    ),
    TimelineEntry(
        2025,
        "saliency_partition",
        "Saliency Partition (counterfactual salient-part reasoning for FGVC)",
        "relation",
    ),
    TimelineEntry(
        2025,
        "late_fusion_transformer",
        "Late Fusion Transformer (multi-view / multi-cue FGVC fusion)",
        "transformer",
    ),
    TimelineEntry(2026, "img_cot", "ImgCoT (compact visual CoT tokens)", "transformer"),
    TimelineEntry(
        2026,
        "refine_rft",
        "ReFine-RFT (reasoning length regulation / cost of thinking)",
        "transformer",
    ),
    TimelineEntry(2026, "iir_vlm", "IIR-VLM (instance-level expert fusion for VLM)", "transformer"),
    TimelineEntry(2026, "fine_r1", "Fine-R1 (CoT-style reasoning tokens for FGVC)", "transformer"),
    TimelineEntry(
        2026, "r2i_distill", "Zooming without Zooming (region-to-image distillation)", "transformer"
    ),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}
