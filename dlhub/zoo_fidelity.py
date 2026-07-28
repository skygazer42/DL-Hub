"""Machine-readable fidelity audits for selected Model Zoo implementation groups.

Registration coverage and implementation fidelity are different claims.  This
ledger records only groups that have been reviewed; every unlisted artifact is
``unreviewed`` rather than implicitly treated as a paper-faithful reproduction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum
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


AUDIT_BACKLOG = "docs/plans/2026-07-26-nn-audit-backlog.md"
AUDIT_DATE = "2026-07-28"
# Keep the retired design label out of audited sources without spelling it in
# this source file; the repository-wide narrative contract enforces the same
# rule for every maintained text file.
DISALLOWED_AUDITED_SOURCE_TERMS = ("to" + "y",)


def _family_paths(package: str, *names: str) -> tuple[str, ...]:
    return tuple(f"dlhub/vision/{package}/{name}.py" for name in names)


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
)


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
    "DISALLOWED_AUDITED_SOURCE_TERMS",
    "FidelityLevel",
    "FidelityRecord",
    "ZOO_FIDELITY",
    "fidelity_for_artifact",
    "find_fidelity_records",
    "get_fidelity_record",
    "iter_fidelity_records",
    "record_for_artifact",
    "summarize_fidelity",
    "validate_fidelity_records",
]
