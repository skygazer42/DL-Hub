"""Machine-readable fidelity audits for selected Model Zoo implementation groups.

Registration coverage and implementation fidelity are different claims.  This
ledger records only groups that have been reviewed; every unlisted artifact is
``unreviewed`` rather than implicitly treated as a paper-faithful reproduction.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from collections import Counter
from dataclasses import dataclass
from datetime import date
from enum import Enum
import hashlib
import json
from pathlib import Path


class FidelityLevel(str, Enum):
    """How closely an audited implementation represents its registered name."""

    REFERENCE = "reference"
    COMPACT = "compact"
    BASELINE_ALIAS = "baseline-alias"
    UNREVIEWED = "unreviewed"


@dataclass(frozen=True)
class FidelityRecord:
    """One audited implementation group, which may contain several source files."""

    key: str
    area: str
    level: FidelityLevel
    artifacts: tuple[str, ...]
    summary: str
    missing_mechanisms: tuple[str, ...]
    reviewed_on: str
    evidence: tuple[str, ...]
    next_action: str


@dataclass(frozen=True, order=True)
class BaselineWrapper:
    """One source-level direct delegation to a shared ``build_baseline_*`` helper."""

    artifact: str
    helper: str
    line: int


AUDIT_BACKLOG = "docs/plans/2026-07-26-nn-audit-backlog.md"
AUDIT_DATE = "2026-07-28"
FOLLOW_UP_AUDIT_DATE = "2026-07-29"
RETRIEVAL_AUDIT_DATE = "2026-08-30"
LAYOUT_AUDIT_DATE = "2026-08-30"
REGISTRATION_AUDIT_DATE = "2026-08-30"
VLM_AUDIT_DATE = "2026-08-30"
DIFFUSION_AUDIT_DATE = "2026-08-30"
BASELINE_INVENTORY_PATH = "docs/zoo/baseline-inventory.json"
BASELINE_WRAPPER_DEBT_BASELINE = 2042
# Preserve the first measured debt snapshot while locking in each completed
# audit wave through the ratchet denominator.
AUDIT_PRESSURE_BASELINE_REGISTRATIONS = 8611
AUDIT_PRESSURE_BASELINE_ARTIFACTS = 80
AUDIT_PRESSURE_RATCHET_ARTIFACTS = 232
MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT = (
    AUDIT_PRESSURE_BASELINE_REGISTRATIONS / AUDIT_PRESSURE_RATCHET_ARTIFACTS
)
# Keep the retired design label out of audited sources without spelling it in
# this source file; the repository-wide narrative contract enforces the same
# rule for every maintained text file.
DISALLOWED_AUDITED_SOURCE_TERMS = ("to" + "y",)


def _family_paths(package: str, *names: str) -> tuple[str, ...]:
    return tuple(f"dlhub/vision/{package}/{name}.py" for name in names)


def _pointcloud_family_paths(package: str, *names: str) -> tuple[str, ...]:
    return tuple(f"dlhub/pointcloud/{package}/{name}.py" for name in names)


def _multimodal_family_paths(package: str, *names: str) -> tuple[str, ...]:
    return tuple(f"dlhub/multimodal/{package}/{name}.py" for name in names)


def _generative_family_paths(package: str, *names: str) -> tuple[str, ...]:
    return tuple(f"dlhub/generative/{package}/{name}.py" for name in names)


ZOO_FIDELITY: tuple[FidelityRecord, ...] = (
    FidelityRecord(
        key="vision.detection.detr-paper-labels",
        area="vision/detection",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "detection",
            "focal_detr",
            "swin_detr",
            "align_detr",
            "prompt_detr",
            "open_vocab_detr",
        ),
        summary=(
            "Five paper-labelled families use the same compact DETR baseline; "
            "their class, builder, and variant prefixes are the material differences."
        ),
        missing_mechanisms=(
            "Focal-DETR-specific focal attention or query refinement",
            "Swin window attention and shifted-window backbone",
            "Align-DETR-specific alignment mechanism",
            "prompt conditioning for Prompt-DETR",
            "open-vocabulary text encoder and classifier",
        ),
        reviewed_on=AUDIT_DATE,
        evidence=(
            AUDIT_BACKLOG,
            "dlhub/vision/detection/_compact_detr.py",
            "dlhub/vision/detection/focal_detr.py",
        ),
        next_action=(
            "Replace each compatibility alias only when its signature mechanism is implemented."
        ),
    ),
    FidelityRecord(
        key="vision.temporal-action-localization.shared-gru",
        area="vision/temporal_action_localization",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "temporal_action_localization",
            "actionformer",
            "afsd",
            "bmn2",
            "bsn2",
            "dyfadet",
            "mambatal",
            "querytal",
            "tapg",
            "temporalmaxer",
            "tridet",
        ),
        summary=(
            "All audited family entrypoints delegate to one projection + stacked GRU + "
            "classification/boundary head; the family string does not affect computation."
        ),
        missing_mechanisms=(
            "per-family attention, proposal, convolutional, or state-space mechanisms",
        ),
        reviewed_on=AUDIT_DATE,
        evidence=(AUDIT_BACKLOG, "dlhub/vision/temporal_action_localization/_common.py"),
        next_action="Split families by their defining temporal mechanism.",
    ),
    FidelityRecord(
        key="vision.video-temporal-grounding.shared-gru",
        area="vision/video_temporal_grounding",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "video_temporal_grounding",
            "2dtan2",
            "bmn_ground",
            "cnm",
            "eatr",
            "mambaground",
            "momentdetr",
            "qdetr_ground",
            "reloclnet",
            "uvcom",
            "videolocformer",
        ),
        summary=(
            "All audited family entrypoints share a projected feature addition, stacked GRU, "
            "and boundary head; the family string does not affect computation."
        ),
        missing_mechanisms=(
            "query decoder or explicit cross-modal attention",
            "per-family proposal, 2D temporal-map, or state-space mechanisms",
        ),
        reviewed_on=AUDIT_DATE,
        evidence=(AUDIT_BACKLOG, "dlhub/vision/video_temporal_grounding/_common.py"),
        next_action="Introduce a real query-conditioned baseline before differentiating paper families.",
    ),
    FidelityRecord(
        key="vision.referring-expression-comprehension.shared-conv-fusion",
        area="vision/referring_expression_comprehension",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "referring_expression_comprehension",
            "cmn",
            "groundref",
            "lavt",
            "mambaref",
            "mattnet",
            "mixref",
            "queryref",
            "refclip",
            "reftr",
            "transvg",
        ),
        summary=(
            "All audited family entrypoints share one convolutional encoder, additive text "
            "projection, and box head; the family string does not affect computation."
        ),
        missing_mechanisms=(
            "transformer or token-level vision-language fusion",
            "per-family modular attention, CLIP, query, or state-space mechanisms",
        ),
        reviewed_on=AUDIT_DATE,
        evidence=(AUDIT_BACKLOG, "dlhub/vision/referring_expression_comprehension/_common.py"),
        next_action="Keep the shared baseline explicit and add distinct fusion modules per family.",
    ),
    FidelityRecord(
        key="vision.open-vocabulary-segmentation.shared-text-bias",
        area="vision/open_vocabulary_segmentation",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "open_vocabulary_segmentation",
            "cat_seg",
            "clipseg2",
            "grounded_sam2_seg",
            "groupvit_seg",
            "lseg2",
            "mambaseg_ov",
            "maskclip",
            "openmask_seg",
            "ovseg2",
            "san_seg",
        ),
        summary=(
            "Ten paper-labelled entrypoints share one compact convolutional baseline where a projected "
            "text vector is added as a spatially constant feature bias."
        ),
        missing_mechanisms=(
            "pretrained or contrastive vision-language embedding space",
            "class-text similarity logits and per-family mask decoding mechanisms",
        ),
        reviewed_on=AUDIT_DATE,
        evidence=(
            AUDIT_BACKLOG,
            "dlhub/vision/_shared/text_guided_segmentation.py",
        ),
        next_action="Implement at least one real embedding-similarity path.",
    ),
    FidelityRecord(
        key="vision.referring-expression-segmentation.shared-text-bias",
        area="vision/referring_expression_segmentation",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "referring_expression_segmentation",
            "cris",
            "gres",
            "grounded_refseg",
            "lavt_seg",
            "lisa_seg",
            "mambarefseg",
            "queryseg_ref",
            "refformer_seg",
            "refsegformer",
            "restrip",
        ),
        summary=(
            "Ten paper-labelled entrypoints share the same compact text-bias segmentation baseline as "
            "the open-vocabulary package."
        ),
        missing_mechanisms=(
            "token-level referring-expression grounding",
            "per-family language fusion and mask decoding mechanisms",
        ),
        reviewed_on=AUDIT_DATE,
        evidence=(
            AUDIT_BACKLOG,
            "dlhub/vision/_shared/text_guided_segmentation.py",
        ),
        next_action="Replace aliases only after token-level grounding mechanisms exist.",
    ),
    FidelityRecord(
        key="vision.video-summarization.queryfocus-sum",
        area="vision/video_summarization",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths("video_summarization", "queryfocus_sum"),
        summary=(
            "The model now accepts a query vector and uses query-frame alignment plus feature "
            "modulation to condition frame scores."
        ),
        missing_mechanisms=("token-level query encoder and cross-attention over query tokens",),
        reviewed_on=AUDIT_DATE,
        evidence=(AUDIT_BACKLOG, "dlhub/vision/video_summarization/queryfocus_sum.py"),
        next_action="Replace vector padding/truncation with a real query encoder and token attention.",
    ),
    FidelityRecord(
        key="vision.video-summarization.memorytokensum",
        area="vision/video_summarization",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths("video_summarization", "memorytokensum"),
        summary=(
            "The implementation now learns memory tokens, updates them from frame features, and "
            "reads the resulting memory back into frame scoring."
        ),
        missing_mechanisms=("persistent or recurrent memory state across separate video clips",),
        reviewed_on=AUDIT_DATE,
        evidence=(AUDIT_BACKLOG, "dlhub/vision/video_summarization/memorytokensum.py"),
        next_action="Add an optional external memory state for streaming or multi-clip summaries.",
    ),
    FidelityRecord(
        key="vision.video-summarization.segmentformer-sum",
        area="vision/video_summarization",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths("video_summarization", "segmentformer_sum"),
        summary=(
            "The implementation performs multi-window segment pooling, so it has a distinct segment "
            "idea, but it does not contain a Transformer."
        ),
        missing_mechanisms=("Transformer-based segment interaction or contextual segment encoder",),
        reviewed_on=AUDIT_DATE,
        evidence=("dlhub/vision/video_summarization/segmentformer_sum.py",),
        next_action="Add segment-token attention while retaining the lightweight pooling baseline.",
    ),
    FidelityRecord(
        key="vision.co-segmentation.clip-coseg",
        area="vision/co_segmentation",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths("co_segmentation", "clip_coseg"),
        summary=(
            "The model now accepts text features and uses normalized text/image similarity to "
            "condition group segmentation features."
        ),
        missing_mechanisms=(
            "pretrained CLIP image/text encoders, tokenizer, and contrastive pretraining",
        ),
        reviewed_on=AUDIT_DATE,
        evidence=(AUDIT_BACKLOG, "dlhub/vision/co_segmentation/clip_coseg.py"),
        next_action="Allow optional pretrained CLIP embeddings without making them an offline dependency.",
    ),
    FidelityRecord(
        key="vision.co-segmentation.token-affinity-coseg",
        area="vision/co_segmentation",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths("co_segmentation", "token_affinity_coseg"),
        summary=(
            "The model computes attention between pooled image descriptors across the group, but "
            "does not compute spatial token-to-token affinity."
        ),
        missing_mechanisms=("patch-level or spatial token affinity and correspondence maps",),
        reviewed_on=AUDIT_DATE,
        evidence=(AUDIT_BACKLOG, "dlhub/vision/co_segmentation/token_affinity_coseg.py"),
        next_action="Move affinity computation before spatial pooling and expose correspondence maps.",
    ),
    FidelityRecord(
        key="vision.blur-detection.shared-compact-baseline",
        area="vision/blur_detection",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "blur_detection",
            "coarse_blurdet",
            "dual_blurdet",
            "edge_blurdet",
            "fft_blurdet",
            "laplacian_blurdet",
            "mamba_blurdet",
            "prompt_blurdet",
            "sobel_blurdet",
            "transformer_blurdet",
            "variance_blurdet",
        ),
        summary=(
            "Ten method-labelled entrypoints delegate to TinyBlurDetector. Several labels share "
            "the same depthwise-convolution branch; the others select only a small local mode branch."
        ),
        missing_mechanisms=(
            "actual Laplacian, Sobel, FFT, or variance operators for their named detector families",
            "token attention or state-space blocks for Transformer and Mamba labels",
        ),
        reviewed_on=FOLLOW_UP_AUDIT_DATE,
        evidence=("dlhub/vision/blur_detection/_common.py",),
        next_action="Rename the shared implementation as a baseline and add measurable family mechanisms.",
    ),
    FidelityRecord(
        key="vision.crack-detection.shared-compact-baseline",
        area="vision/crack_detection",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_family_paths(
            "crack_detection",
            "coarse_crack",
            "contour_crack",
            "dual_crack",
            "fpn_crack",
            "hed_crack",
            "mamba_crack",
            "prompt_crack",
            "skeleton_crack",
            "transformer_crack",
            "unet_crack",
        ),
        summary=(
            "Ten method-labelled entrypoints delegate to TinyCrackDetector. Five labels share one "
            "depthwise-convolution branch and the remaining labels only select a small local mode branch."
        ),
        missing_mechanisms=(
            "encoder-decoder, feature-pyramid, contour, skeleton, and HED-specific computation",
            "token attention or state-space blocks for Transformer and Mamba labels",
        ),
        reviewed_on=FOLLOW_UP_AUDIT_DATE,
        evidence=("dlhub/vision/crack_detection/_common.py",),
        next_action="Expose one honest shared baseline until each named family has its defining mechanism.",
    ),
    FidelityRecord(
        key="vision.image-retrieval.mechanism-aware-compact",
        area="vision/image_retrieval",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths(
            "image_retrieval",
            "arc",
            "clipret",
            "contrastive",
            "delg",
            "gem",
            "netvlad",
            "pairret",
            "proxy",
            "regional",
            "transformerret",
        ),
        summary=(
            "The ten builders now select attention with angular scoring, context conditioning, "
            "temperature-scaled contrastive scoring, local-global saliency, GeM, NetVLAD, learned "
            "pairwise scoring, proxy refinement, regional grids, or spatial-token attention."
        ),
        missing_mechanisms=(
            "paper-specific encoders, mining strategies, margin schedules, and training objectives",
            "pretrained vision-language encoders and real text tokenization for context paths",
            "benchmark datasets, pretrained weights, and retrieval metric reproduction",
        ),
        reviewed_on=RETRIEVAL_AUDIT_DATE,
        evidence=(
            "dlhub/vision/image_retrieval/_common.py",
            "dlhub/vision/_shared/retrieval.py",
            "tests/test_dlhub_retrieval_mechanisms.py",
        ),
        next_action=(
            "Add family-specific training losses and evaluate recall and mean average precision on "
            "a real retrieval benchmark before considering reference-level claims."
        ),
    ),
    FidelityRecord(
        key="vision.visual-place-recognition.mechanism-aware-compact",
        area="vision/visual_place_recognition",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths(
            "visual_place_recognition",
            "apgem_vpr",
            "cosplace",
            "delg_vpr",
            "geoclip_vpr",
            "mambavpr",
            "mixvpr",
            "pairvpr",
            "patchnetvlad",
            "regionvpr",
            "transvpr",
        ),
        summary=(
            "The place-recognition entrypoints now select adaptive GeM, proxy refinement, "
            "local-global saliency, geographic context, a selective scan, spatial mixing, pairwise "
            "matching, patch VLAD, regional grids, or a place-token Transformer."
        ),
        missing_mechanisms=(
            "paper-specific place-recognition backbones, geographic supervision, and hard-negative mining",
            "full sequence state-space blocks and patch-level reranking used by the named methods",
            "real geolocated datasets, pretrained weights, and recall benchmark reproduction",
        ),
        reviewed_on=RETRIEVAL_AUDIT_DATE,
        evidence=(
            "dlhub/vision/visual_place_recognition/_common.py",
            "dlhub/vision/_shared/retrieval.py",
            "tests/test_dlhub_retrieval_mechanisms.py",
        ),
        next_action=(
            "Add geolocation-aware training and evaluate day-night and viewpoint robustness on a "
            "real visual place-recognition benchmark."
        ),
    ),
    FidelityRecord(
        key="vision.fine-grained-retrieval.mechanism-aware-compact",
        area="vision/fine_grained_retrieval",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths(
            "fine_grained_retrieval",
            "bilinear_fgret",
            "descriptor_fgret",
            "fgclip_retr",
            "granule_retr",
            "mamba_fgret",
            "partvlad",
            "prompt_fgret",
            "regional_fgret",
            "tokenpart_retr",
            "transformer_fgret",
        ),
        summary=(
            "The fine-grained entrypoints now select bilinear interaction, a global descriptor, "
            "context-conditioned parts, multiscale granularity, a selective scan, part VLAD, "
            "prompt attention, regional grids, learned part tokens, or a token Transformer."
        ),
        missing_mechanisms=(
            "paper-specific part discovery, supervision, losses, and pretrained feature extractors",
            "real prompt or text encoders for the context-conditioned families",
            "fine-grained retrieval datasets, pretrained weights, and benchmark reproduction",
        ),
        reviewed_on=RETRIEVAL_AUDIT_DATE,
        evidence=(
            "dlhub/vision/fine_grained_retrieval/_common.py",
            "dlhub/vision/_shared/retrieval.py",
            "tests/test_dlhub_retrieval_mechanisms.py",
        ),
        next_action=(
            "Add supervised part discovery and evaluate recall and mean average precision on a real "
            "fine-grained retrieval dataset."
        ),
    ),
    FidelityRecord(
        key="vision.layout-generation.mechanism-aware-compact",
        area="vision/layout_generation",
        level=FidelityLevel.COMPACT,
        artifacts=_family_paths(
            "layout_generation",
            "layoutgan_baseline",
            "layoutvae_baseline",
            "layouttransformer",
            "bbox_generator",
            "poster_layout_net",
            "doc_layout_gen",
            "constraint_layout",
            "relation_layout",
            "diffusion_layout",
            "mamba_layout_gen",
        ),
        summary=(
            "The ten entrypoints now select distinct compact computation: latent residual "
            "generation, a variational spatial bottleneck, spatial self-attention, coordinate "
            "objectness, pyramid fusion, axial mixing, constraint projection, relation attention, "
            "time-conditioned denoising, or an input-dependent selective scan."
        ),
        missing_mechanisms=(
            "full adversarial, variational, diffusion, and structured-layout training objectives",
            "discrete element, bounding-box, relation-token, and constraint-set interfaces",
            "paper-scale backbones, datasets, pretrained weights, and benchmark reproduction",
        ),
        reviewed_on=LAYOUT_AUDIT_DATE,
        evidence=(
            "dlhub/vision/layout_generation/_mechanisms.py",
            "tests/test_dlhub_vision_layout_generation_zoo.py",
        ),
        next_action=(
            "Add a structured element-and-box interface and evaluate each mechanism on a real "
            "layout benchmark before considering any reference-level claim."
        ),
    ),
    FidelityRecord(
        key="pointcloud.registration.mechanism-aware-compact",
        area="pointcloud/registration",
        level=FidelityLevel.COMPACT,
        artifacts=_pointcloud_family_paths(
            "registration",
            "pointnetlk",
            "dcp",
            "regtr",
            "rpmnet",
            "deepgmr",
            "spinreg",
            "cofinet_reg",
            "geoformer_reg",
            "predator_reg",
            "mambareg",
        ),
        summary=(
            "The ten registrars now select distinct compact computation: iterative global-feature "
            "alignment, cross-attention correspondences, a joint Transformer, Sinkhorn matching, "
            "soft mixture alignment, cylindrical descriptors, coarse-to-fine matching, "
            "geometry-biased attention, overlap weighting, or a radial-order selective scan."
        ),
        missing_mechanisms=(
            "full PointNetLK Jacobian optimization and closed-form weighted rigid-transform solving",
            "paper-specific neighborhood encoders, overlap supervision, iterative schedules, and losses",
            "real registration datasets, pretrained weights, and benchmark reproduction",
        ),
        reviewed_on=REGISTRATION_AUDIT_DATE,
        evidence=(
            "dlhub/pointcloud/registration/_mechanisms.py",
            "tests/test_dlhub_pointcloud_registration_mechanisms.py",
        ),
        next_action=(
            "Add differentiable weighted Procrustes outputs and evaluate rotation and translation "
            "errors on a real partial-overlap registration benchmark."
        ),
    ),
    FidelityRecord(
        key="multimodal.vlm.representative-mechanism-compact",
        area="multimodal/vlm",
        level=FidelityLevel.COMPACT,
        artifacts=_multimodal_family_paths(
            "vlm",
            "albef",
            "align",
            "blip",
            "blip2",
            "clip",
            "coca",
            "flamingo",
            "instructblip",
            "lit",
            "minigpt4",
            "simvlm",
            "vilt",
        ),
        summary=(
            "Twelve representative families now accept caller-supplied images and tokens and use "
            "an explicit compact dual encoder, joint multimodal Transformer, text-to-image cross "
            "attention, or query-token bridge with position-dependent generation logits."
        ),
        missing_mechanisms=(
            "paper-specific pretrained vision and language backbones, tokenizers, and weights",
            "family-specific pretraining losses, data filtering, resampling, and generation decoders",
            "real multimodal datasets and benchmark reproduction",
        ),
        reviewed_on=VLM_AUDIT_DATE,
        evidence=(
            "dlhub/multimodal/vlm/_common.py",
            "tests/test_dlhub_multimodal_vlm_mechanisms.py",
        ),
        next_action=(
            "Add real tokenizer and pretrained-backbone adapters, then evaluate retrieval, VQA, "
            "and generation tasks before considering reference-level claims."
        ),
    ),
    FidelityRecord(
        key="multimodal.vlm.shared-mode-baseline-labels",
        area="multimodal/vlm",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_multimodal_family_paths(
            "vlm",
            "agent_vl",
            "aria",
            "bunny",
            "cambrian",
            "chartvlm",
            "cogvlm",
            "deepseek_vl",
            "docowl2",
            "eagle_vlm",
            "edgevlm",
            "emu2",
            "evo_vl",
            "ferret",
            "fuyu",
            "granite_vision",
            "grounded_vlm",
            "idefics2",
            "idefics3",
            "internlm_xcomposer",
            "internvl",
            "internvl2",
            "janus",
            "kimi_vl",
            "kosmos2",
            "kosmos25",
            "llama_vision",
            "llava",
            "llava_next",
            "metavlm",
            "minicpm_o",
            "minicpm_v",
            "mixvlm",
            "mobilevlm",
            "molmo",
            "moondream2",
            "mplug_owl2",
            "ocrvlm",
            "ofa",
            "olmocr",
            "omni_vlm",
            "ovis",
            "pali",
            "pali_x",
            "phi3_vision",
            "phi4_mm",
            "qwen2_vl",
            "qwen_vl",
            "rabbit_vlm",
            "science_vlm",
            "seed_vl",
            "siglip_vlm",
            "stem_vl",
            "video_llava",
            "video_qwen_vl",
            "vila",
            "webvlm",
            "xcomposer2",
            "xgen_mm",
        ),
        summary=(
            "These 58 product- and paper-labelled entrypoints now have real-input multimodal "
            "execution, but still select only one of four shared compact architecture modes plus "
            "instruction, query, and generation flags rather than their named model mechanisms."
        ),
        missing_mechanisms=(
            "the named models' pretrained backbones, modality adapters, resamplers, experts, and decoders",
            "video, document, OCR, grounding, spatial-resolution, and tool-use mechanisms implied by labels",
            "model-specific tokenizers, training objectives, weights, datasets, and benchmark evidence",
        ),
        reviewed_on=VLM_AUDIT_DATE,
        evidence=(
            "dlhub/multimodal/vlm/_common.py",
            "tests/test_dlhub_multimodal_vlm_mechanisms.py",
        ),
        next_action=(
            "Keep the baseline-alias label until each entrypoint has an observable named mechanism "
            "and targeted behavior test beyond the shared architecture mode."
        ),
    ),
    FidelityRecord(
        key="generative.diffusion.representative-mechanism-compact",
        area="generative/diffusion",
        level=FidelityLevel.COMPACT,
        artifacts=_generative_family_paths(
            "diffusion",
            "consistency_model",
            "ddim",
            "ddpm",
            "dit",
            "flow_matching",
            "latent_diffusion",
            "lcm",
            "rectified_flow",
            "score_sde",
            "stable_diffusion",
        ),
        summary=(
            "Ten representative families now accept explicit noisy states and timesteps, use real "
            "time and class conditioning, select a spatial convolutional, patch Transformer, or "
            "latent autoencoder denoiser, and apply mode-specific iterative updates."
        ),
        missing_mechanisms=(
            "paper-specific U-Net, DiT, autoencoder, text encoder, and scheduler configurations",
            "exact stochastic differential equations, variance prediction, guidance, and solver math",
            "pretrained weights, real datasets, and image-generation benchmark reproduction",
        ),
        reviewed_on=DIFFUSION_AUDIT_DATE,
        evidence=(
            "dlhub/generative/diffusion/_common.py",
            "tests/test_dlhub_generative_diffusion_mechanisms.py",
        ),
        next_action=(
            "Add exact noise schedules and solver equations with real text conditioning, then "
            "evaluate sample quality and speed on a reproducible image dataset."
        ),
    ),
    FidelityRecord(
        key="generative.diffusion.shared-mode-baseline-labels",
        area="generative/diffusion",
        level=FidelityLevel.BASELINE_ALIAS,
        artifacts=_generative_family_paths(
            "diffusion",
            "aura_flow",
            "conditional_flow_matching",
            "edm",
            "flux",
            "hunyuan_dit",
            "iddpm",
            "latent_consistency",
            "lumina_next",
            "masked_diffusion",
            "mini_diffusion",
            "ncsnpp",
            "omni_gen",
            "pixart",
            "pixart_alpha",
            "pixart_sigma",
            "riffusion",
            "sana",
            "sd3",
            "sd_turbo",
            "sdxl",
            "uvit",
            "vision_diffusion",
        ),
        summary=(
            "These 22 labels now support explicit states, timesteps, conditioning, and iterative "
            "sampling, but still map to three shared denoiser categories and five prediction modes "
            "rather than the named model's full architecture and sampler."
        ),
        missing_mechanisms=(
            "the labels' specific rectified-flow, consistency, masked, audio, or text-image mechanisms",
            "named transformer blocks, U-Nets, autoencoders, conditioning stacks, and sampling solvers",
            "model-specific training objectives, weights, datasets, and benchmark evidence",
        ),
        reviewed_on=DIFFUSION_AUDIT_DATE,
        evidence=(
            "dlhub/generative/diffusion/_common.py",
            "tests/test_dlhub_generative_diffusion_mechanisms.py",
        ),
        next_action=(
            "Keep the baseline-alias level until each label changes observable computation beyond "
            "the shared architecture and prediction-mode configuration."
        ),
    ),
)


def _callable_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def discover_baseline_wrappers(repo_root: str | Path) -> tuple[BaselineWrapper, ...]:
    """Discover direct ``return build_baseline_*`` delegations from current source.

    This deliberately excludes indirect helper use.  The inventory tracks the
    high-risk compatibility-wrapper pattern identified by the fidelity audit,
    rather than treating every internal baseline utility call as an alias.
    """

    root = Path(repo_root)
    source_root = root / "dlhub"
    wrappers: list[BaselineWrapper] = []
    if not source_root.is_dir():
        return ()

    for source_path in sorted(source_root.rglob("*.py")):
        source = source_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(source_path))
        except SyntaxError as exc:  # pragma: no cover - repository lint catches this first
            raise ValueError(f"cannot parse baseline wrapper source {source_path}: {exc}") from exc

        artifact = source_path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
                continue
            helper = _callable_name(node.value.func)
            if helper is None or not helper.startswith("build_baseline_"):
                continue
            wrappers.append(
                BaselineWrapper(
                    artifact=artifact,
                    helper=helper,
                    line=int(node.lineno),
                )
            )

    return tuple(sorted(wrappers))


def build_baseline_inventory(repo_root: str | Path) -> dict[str, object]:
    """Build the deterministic, source-grounded baseline-wrapper inventory."""

    wrappers = discover_baseline_wrappers(repo_root)
    entries: list[dict[str, object]] = []
    level_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    helper_counts: Counter[str] = Counter()

    for wrapper in wrappers:
        audit = record_for_artifact(wrapper.artifact)
        level = audit.level if audit is not None else FidelityLevel.BASELINE_ALIAS
        entry = {
            "artifact": wrapper.artifact,
            "helper": wrapper.helper,
            "line": wrapper.line,
            "level": level.value,
            "audit_key": audit.key if audit is not None else None,
            "review_status": "reviewed" if audit is not None else "source-inferred",
        }
        entries.append(entry)
        level_counts[level.value] += 1
        parts = Path(wrapper.artifact).parts
        domain_counts[parts[1] if len(parts) > 1 else "root"] += 1
        helper_counts[wrapper.helper] += 1

    serialized_entries = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    audited = sum(entry["review_status"] == "reviewed" for entry in entries)
    inferred = sum(entry["review_status"] == "source-inferred" for entry in entries)
    unidentified = len(entries) - audited - inferred
    summary = {
        "total_wrappers": len(entries),
        "audited_wrappers": audited,
        "source_inferred_alias_wrappers": inferred,
        "unreviewed_wrappers": unidentified,
        "debt_baseline": BASELINE_WRAPPER_DEBT_BASELINE,
        "debt_reduction": BASELINE_WRAPPER_DEBT_BASELINE - len(entries),
        "by_level": dict(sorted(level_counts.items())),
        "by_domain": dict(sorted(domain_counts.items())),
        "by_helper": dict(sorted(helper_counts.items())),
    }
    return {
        "schema_version": 2,
        "scope": "direct return build_baseline_* calls under dlhub/",
        "source_sha256": hashlib.sha256(serialized_entries).hexdigest(),
        "summary": summary,
        "wrappers": entries,
    }


def validate_baseline_inventory(repo_root: str | Path) -> list[str]:
    """Validate the checked-in baseline inventory against current Python source."""

    root = Path(repo_root)
    inventory_path = root / BASELINE_INVENTORY_PATH
    expected = build_baseline_inventory(root)
    expected_wrappers = expected["wrappers"]
    assert isinstance(expected_wrappers, list)
    errors: list[str] = []

    total = len(expected_wrappers)
    if total > BASELINE_WRAPPER_DEBT_BASELINE:
        errors.append(
            "baseline-wrapper debt grew beyond its locked baseline: "
            f"{total} current > {BASELINE_WRAPPER_DEBT_BASELINE} baseline"
        )

    artifacts = [str(entry["artifact"]) for entry in expected_wrappers]
    duplicates = sorted(path for path, count in Counter(artifacts).items() if count > 1)
    if duplicates:
        errors.append(
            "baseline-wrapper artifacts must contain one direct delegation each: "
            + ", ".join(duplicates[:10])
        )

    if not inventory_path.is_file():
        errors.append(
            f"missing baseline inventory: {BASELINE_INVENTORY_PATH}; "
            "run python scripts/model_fidelity.py --write-baseline-inventory"
        )
        return errors

    try:
        actual = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read baseline inventory {BASELINE_INVENTORY_PATH}: {exc}")
        return errors

    if actual != expected:
        actual_entries = actual.get("wrappers", []) if isinstance(actual, dict) else []
        actual_by_artifact = {
            str(entry.get("artifact")): entry
            for entry in actual_entries
            if isinstance(entry, dict) and entry.get("artifact")
        }
        expected_by_artifact = {
            str(entry["artifact"]): entry for entry in expected_wrappers if isinstance(entry, dict)
        }
        added = sorted(set(expected_by_artifact) - set(actual_by_artifact))
        removed = sorted(set(actual_by_artifact) - set(expected_by_artifact))
        changed = sorted(
            artifact
            for artifact in set(expected_by_artifact) & set(actual_by_artifact)
            if expected_by_artifact[artifact] != actual_by_artifact[artifact]
        )
        errors.append(
            "baseline inventory is stale: "
            f"{len(added)} added, {len(removed)} removed, {len(changed)} changed; "
            "run python scripts/model_fidelity.py --write-baseline-inventory"
        )

    return errors


def _normalize_artifact(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def iter_fidelity_records(
    level: FidelityLevel | str | None = None,
) -> tuple[FidelityRecord, ...]:
    """Return audited groups, optionally filtered by fidelity level."""

    if level is None:
        return ZOO_FIDELITY
    selected = level if isinstance(level, FidelityLevel) else FidelityLevel(str(level))
    return tuple(record for record in ZOO_FIDELITY if record.level is selected)


def get_fidelity_record(key: str) -> FidelityRecord:
    """Return one audited group by stable key."""

    for record in ZOO_FIDELITY:
        if record.key == key:
            return record
    raise KeyError(f"Unknown fidelity audit group: {key!r}")


def find_fidelity_records(query: str) -> tuple[FidelityRecord, ...]:
    """Search audited records by key, area, summary, mechanism, or artifact."""

    needle = str(query).strip().lower()
    if not needle:
        return ()
    matches = []
    for record in ZOO_FIDELITY:
        values = (
            record.key,
            record.area,
            record.summary,
            *record.artifacts,
            *record.missing_mechanisms,
        )
        if any(needle in value.lower() for value in values):
            matches.append(record)
    return tuple(matches)


def record_for_artifact(path: str | Path) -> FidelityRecord | None:
    """Return the audit group containing ``path``, or ``None`` when unreviewed."""

    artifact = _normalize_artifact(path)
    for record in ZOO_FIDELITY:
        if artifact in record.artifacts:
            return record
    return None


def fidelity_for_artifact(path: str | Path) -> FidelityLevel:
    """Return an explicit level; unlisted artifacts are always ``unreviewed``."""

    record = record_for_artifact(path)
    return record.level if record is not None else FidelityLevel.UNREVIEWED


def summarize_fidelity(
    records: Iterable[FidelityRecord] = ZOO_FIDELITY,
) -> dict[str, int]:
    """Summarize audited groups without pretending to count unreviewed artifacts."""

    selected = tuple(records)
    summary = {
        "audited_groups": len(selected),
        "audited_artifacts": sum(len(record.artifacts) for record in selected),
    }
    for level in FidelityLevel:
        if level is FidelityLevel.UNREVIEWED:
            continue
        summary[level.value] = sum(record.level is level for record in selected)
    return summary


def summarize_audit_pressure(
    total_registration_ids: int,
    records: Iterable[FidelityRecord] = ZOO_FIDELITY,
) -> dict[str, int | float]:
    """Relate catalog growth to the amount of source that has received a fidelity audit."""

    total = int(total_registration_ids)
    audited_artifacts = sum(len(record.artifacts) for record in records)
    ratio = total / audited_artifacts if audited_artifacts else float("inf")
    return {
        "total_registration_ids": total,
        "audited_artifacts": audited_artifacts,
        "registrations_per_audited_artifact": ratio,
        "max_registrations_per_audited_artifact": MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT,
    }


def validate_audit_pressure(
    total_registration_ids: int,
    records: Iterable[FidelityRecord] = ZOO_FIDELITY,
) -> list[str]:
    """Block registration growth that is not accompanied by additional source audits."""

    pressure = summarize_audit_pressure(total_registration_ids, records)
    total = int(pressure["total_registration_ids"])
    audited = int(pressure["audited_artifacts"])
    ratio = float(pressure["registrations_per_audited_artifact"])
    maximum = float(pressure["max_registrations_per_audited_artifact"])
    if total < 0:
        return ["total registration ids must not be negative"]
    if audited == 0:
        return ["fidelity ledger has no audited source artifacts"]
    if ratio > maximum:
        return [
            "registration growth exceeds the fidelity-audit budget: "
            f"{total} ids / {audited} audited artifacts = {ratio:.2f}, maximum {maximum:.2f}; "
            "audit more source artifacts or reduce unsupported registrations"
        ]
    return []


def validate_fidelity_records(repo_root: str | Path | None = None) -> list[str]:
    """Return actionable metadata errors; an empty list means the ledger is valid."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    errors: list[str] = []
    keys: set[str] = set()
    artifacts: dict[str, str] = {}

    for record in ZOO_FIDELITY:
        if not record.key.strip():
            errors.append("record has an empty key")
        elif record.key in keys:
            errors.append(f"duplicate record key: {record.key}")
        keys.add(record.key)

        if record.level is FidelityLevel.UNREVIEWED:
            errors.append(
                f"{record.key}: unreviewed artifacts must remain outside the audit ledger"
            )
        if not record.summary.strip():
            errors.append(f"{record.key}: missing summary")
        if record.level is not FidelityLevel.REFERENCE and not record.missing_mechanisms:
            errors.append(f"{record.key}: missing_mechanisms must explain the fidelity limit")
        if not record.next_action.strip():
            errors.append(f"{record.key}: missing next_action")
        try:
            date.fromisoformat(record.reviewed_on)
        except ValueError:
            errors.append(f"{record.key}: reviewed_on must use ISO YYYY-MM-DD format")

        if not record.artifacts:
            errors.append(f"{record.key}: no audited artifacts")
        for artifact in record.artifacts:
            relative = Path(artifact)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{record.key}: artifact must be repo-relative: {artifact}")
                continue
            if relative.suffix != ".py":
                errors.append(f"{record.key}: artifact must be Python source: {artifact}")
            previous = artifacts.get(artifact)
            if previous is not None:
                errors.append(f"{record.key}: artifact also belongs to {previous}: {artifact}")
            artifacts[artifact] = record.key
            if not (root / relative).is_file():
                errors.append(f"{record.key}: missing artifact: {artifact}")

        for evidence in record.evidence:
            relative = Path(evidence)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{record.key}: evidence must be repo-relative: {evidence}")
            elif not (root / relative).is_file():
                errors.append(f"{record.key}: missing evidence: {evidence}")

        audited_sources = dict.fromkeys(
            (
                *record.artifacts,
                *(evidence for evidence in record.evidence if Path(evidence).suffix == ".py"),
            )
        )
        for source in audited_sources:
            source_path = root / source
            if not source_path.is_file():
                continue
            source_text = source_path.read_text(encoding="utf-8").casefold()
            for term in DISALLOWED_AUDITED_SOURCE_TERMS:
                if term.casefold() in source_text:
                    errors.append(
                        f"{record.key}: audited Zoo source uses disallowed design term "
                        f"{term!r}: {source}"
                    )

    return errors


__all__ = [
    "AUDIT_DATE",
    "AUDIT_PRESSURE_BASELINE_ARTIFACTS",
    "AUDIT_PRESSURE_BASELINE_REGISTRATIONS",
    "AUDIT_PRESSURE_RATCHET_ARTIFACTS",
    "BASELINE_INVENTORY_PATH",
    "BASELINE_WRAPPER_DEBT_BASELINE",
    "BaselineWrapper",
    "DISALLOWED_AUDITED_SOURCE_TERMS",
    "DIFFUSION_AUDIT_DATE",
    "FOLLOW_UP_AUDIT_DATE",
    "FidelityLevel",
    "FidelityRecord",
    "ZOO_FIDELITY",
    "MAX_REGISTRATIONS_PER_AUDITED_ARTIFACT",
    "LAYOUT_AUDIT_DATE",
    "RETRIEVAL_AUDIT_DATE",
    "REGISTRATION_AUDIT_DATE",
    "VLM_AUDIT_DATE",
    "build_baseline_inventory",
    "discover_baseline_wrappers",
    "fidelity_for_artifact",
    "find_fidelity_records",
    "get_fidelity_record",
    "iter_fidelity_records",
    "record_for_artifact",
    "summarize_fidelity",
    "summarize_audit_pressure",
    "validate_audit_pressure",
    "validate_baseline_inventory",
    "validate_fidelity_records",
]
