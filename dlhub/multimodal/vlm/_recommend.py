from __future__ import annotations

from dataclasses import dataclass

from ._timeline import entries


@dataclass(frozen=True)
class ProfileSpec:
    key: str
    title: str
    summary: str
    preferred_groups: tuple[str, ...]
    preferred_families: tuple[str, ...]
    discouraged_groups: tuple[str, ...] = ()
    modern_bias: float = 0.0


@dataclass(frozen=True)
class Recommendation:
    profile: str
    arch_id: str
    family: str
    group: str
    year: int | None
    score: float
    reason: str


_PROFILES: dict[str, ProfileSpec] = {
    "balanced": ProfileSpec(
        key="balanced",
        title="Balanced VLM Starter",
        summary="Mix retrieval, fusion, and multimodal LLM families.",
        preferred_groups=(
            "single_stream",
            "dual_encoder",
            "fusion_encoder_decoder",
            "multimodal_llm",
        ),
        preferred_families=("clip", "lit", "albef", "blip", "pali", "blip2", "qwen_vl", "llava"),
        modern_bias=0.2,
    ),
    "retrieval": ProfileSpec(
        key="retrieval",
        title="Retrieval and Alignment",
        summary="Prefer contrastive image-text alignment families.",
        preferred_groups=("dual_encoder", "single_stream"),
        preferred_families=("clip", "lit", "align", "albef", "vilt"),
        discouraged_groups=("multimodal_llm",),
        modern_bias=0.1,
    ),
    "captioning": ProfileSpec(
        key="captioning",
        title="Captioning and Generation",
        summary="Prefer fusion and generative multimodal families.",
        preferred_groups=("fusion_encoder_decoder", "multimodal_llm"),
        preferred_families=("simvlm", "blip", "ofa", "pali", "coca", "pali_x", "flamingo", "blip2"),
        modern_bias=0.2,
    ),
    "instruction": ProfileSpec(
        key="instruction",
        title="Instruction-Tuned VLM",
        summary="Prefer bridge-based multimodal LLMs and instruction-aware families.",
        preferred_groups=("multimodal_llm", "fusion_encoder_decoder"),
        preferred_families=("qwen_vl", "llava", "cogvlm", "mplug_owl2", "instructblip", "minigpt4", "blip2", "kosmos2", "flamingo"),
        modern_bias=0.35,
    ),
    "lightweight": ProfileSpec(
        key="lightweight",
        title="Lightweight VLM",
        summary="Prefer earlier compact alignment and fusion baselines.",
        preferred_groups=("single_stream", "dual_encoder", "fusion_encoder_decoder"),
        preferred_families=("vilt", "clip", "lit", "align", "albef", "simvlm", "blip"),
        discouraged_groups=("multimodal_llm",),
        modern_bias=0.05,
    ),
}


def list_profiles() -> list[ProfileSpec]:
    return [*(_PROFILES[key] for key in sorted(_PROFILES))]


def _family_year_bonus(year: int | None, *, modern_bias: float) -> float:
    if year is None:
        return 0.0
    span = max(0.0, min(1.0, (float(year) - 2021.0) / 4.0))
    return float(modern_bias) * span


def _family_priority_bonus(spec: ProfileSpec, family: str) -> float:
    if family not in spec.preferred_families:
        return 0.0
    rank = spec.preferred_families.index(family)
    return max(0.0, 0.5 - 0.05 * float(rank))


def _score_family(spec: ProfileSpec, *, family: str, group: str, year: int | None) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    if group in spec.preferred_groups:
        score += 3.0
        reasons.append(f"group={group}")
    if group in spec.discouraged_groups:
        score -= 1.5
        reasons.append(f"discourage={group}")
    if family in spec.preferred_families:
        score += 4.0 + _family_priority_bonus(spec, family)
        reasons.append(f"family={family}")
    score += _family_year_bonus(year, modern_bias=spec.modern_bias)
    if year is not None and spec.modern_bias > 0:
        reasons.append(f"year={year}")
    return score, ", ".join(reasons)


def recommend_arches(
    profile: str,
    *,
    variant: str = "tiny",
    top_k: int = 10,
) -> list[Recommendation]:
    profile_key = str(profile).strip().lower()
    spec = _PROFILES.get(profile_key)
    if spec is None:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown VLM recommendation profile: {profile!r}. Supported: {supported}")

    variant_name = str(variant).strip().lower()
    if variant_name not in {"tiny", "small", "base"}:
        raise ValueError(f"Unknown variant: {variant!r}. Supported: tiny, small, base")
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    from dlhub.multimodal.vlm_zoo import list_local_arches

    arches_set = set(list_local_arches())
    candidates: list[Recommendation] = []
    for entry in entries():
        arch_id = f"vlm:{entry.family}_{variant_name}"
        if arch_id not in arches_set:
            continue
        score, reason = _score_family(spec, family=entry.family, group=entry.group, year=entry.year)
        candidates.append(
            Recommendation(
                profile=profile_key,
                arch_id=arch_id,
                family=entry.family,
                group=entry.group,
                year=entry.year,
                score=score,
                reason=reason,
            )
        )

    candidates.sort(
        key=lambda rec: (
            -float(rec.score),
            9999 if rec.year is None else -int(rec.year),
            rec.family,
        )
    )
    return candidates[: int(top_k)]


__all__ = ["ProfileSpec", "Recommendation", "list_profiles", "recommend_arches"]
