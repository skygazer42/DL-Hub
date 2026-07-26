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
        title="Balanced Starter",
        summary="兼顾速度与鲁棒性的默认入口，先跑通再细分。",
        preferred_groups=("online_association", "joint_det_embed", "query_transformer"),
        preferred_families=(
            "bytetrack",
            "fairmot",
            "trackformer",
            "motr",
            "bot_sort",
            "motdt",
            "motip",
            "masktrack_rcnn",
            "motrv2",
        ),
        discouraged_groups=("global_optimization",),
        modern_bias=0.4,
    ),
    "realtime": ProfileSpec(
        key="realtime",
        title="Real-Time Priority",
        summary="优先在线关联家族，强调推理速度与部署简洁性。",
        preferred_groups=("online_association",),
        preferred_families=(
            "sort",
            "iou_tracker",
            "bytetrack",
            "ocsort",
            "bot_sort",
            "deepsort",
            "motdt",
            "uav_sort",
        ),
        discouraged_groups=("global_optimization", "probabilistic_filtering"),
        modern_bias=0.2,
    ),
    "occlusion": ProfileSpec(
        key="occlusion",
        title="Occlusion Robustness",
        summary="优先重关联和遮挡恢复能力较强的方向。",
        preferred_groups=("online_association", "joint_det_embed", "query_transformer"),
        preferred_families=(
            "strongsort",
            "strongsort_pp",
            "deep_ocsort",
            "trackformer",
            "memotr",
            "qdtrack",
            "crowdsort",
            "relationtrack",
            "sparse_reid_track",
            "tokentrack",
        ),
        discouraged_groups=("global_optimization",),
        modern_bias=0.5,
    ),
    "long_horizon": ProfileSpec(
        key="long_horizon",
        title="Long-Horizon Tracking",
        summary="面向长时序稳定性与全局一致性。",
        preferred_groups=("query_transformer", "probabilistic_filtering", "global_optimization"),
        preferred_families=(
            "motr",
            "memotr",
            "global_hypothesis_bank",
            "trackletnet",
            "network_flow",
            "streamtrack",
            "rbmht",
            "graph_stitching",
            "temporal_clique",
            "variational_mht",
        ),
        modern_bias=0.4,
    ),
    "low_compute": ProfileSpec(
        key="low_compute",
        title="Low Compute / Edge",
        summary="面向低算力部署，优先传统或轻量在线关联方案。",
        preferred_groups=("online_association", "probabilistic_filtering"),
        preferred_families=(
            "sort",
            "iou_tracker",
            "v_iou",
            "deepsort",
            "jpda",
            "mht",
            "camshift_sort",
            "gibbs_jpda",
        ),
        discouraged_groups=("query_transformer",),
        modern_bias=0.0,
    ),
    "transformer": ProfileSpec(
        key="transformer",
        title="Transformer MOT",
        summary="聚焦 query/transformer 技术栈。",
        preferred_groups=("query_transformer",),
        preferred_families=(
            "transtrack",
            "trackformer",
            "motr",
            "memotr",
            "sparsetrack",
            "unicorn",
            "motip",
            "deformtrack",
            "streamtrack",
            "relationformer_track",
            "stq_track",
            "motrv2",
            "qdetr_track",
            "track_deformer",
            "tokentrack",
        ),
        modern_bias=0.8,
    ),
    "global_opt": ProfileSpec(
        key="global_opt",
        title="Global Optimization",
        summary="聚焦图优化与全局关联。",
        preferred_groups=("global_optimization",),
        preferred_families=(
            "network_flow",
            "k_shortest_path",
            "lifted_multicut",
            "gnn_assoc",
            "min_cost_flow",
            "lagrangian_assoc",
            "graph_cut_track",
            "mwis_assoc",
            "benders_flow",
            "temporal_clique",
            "graph_stitching",
        ),
        modern_bias=0.1,
    ),
    "probabilistic": ProfileSpec(
        key="probabilistic",
        title="Probabilistic Filtering",
        summary="聚焦 MHT/JPDA/RFS 等概率滤波与假设管理。",
        preferred_groups=("probabilistic_filtering",),
        preferred_families=(
            "mht",
            "jpda",
            "glmb_lmb",
            "pmbm_gmphd",
            "global_hypothesis_bank",
            "particle_filter_bank",
            "rbmht",
            "phd_lmb",
            "gibbs_jpda",
            "bernoulli_mixture_track",
            "variational_mht",
        ),
        modern_bias=0.2,
    ),
}


def list_profiles() -> list[ProfileSpec]:
    return [*(_PROFILES[k] for k in sorted(_PROFILES))]


def _family_year_bonus(year: int | None, *, modern_bias: float) -> float:
    if year is None:
        return 0.0
    # Normalize around recent MOT development years to avoid overpowering group/family priors.
    span = max(0.0, min(1.0, (float(year) - 2014.0) / 12.0))
    return float(modern_bias) * span


def _score_family(
    spec: ProfileSpec, *, family: str, group: str, year: int | None
) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []

    if group in spec.preferred_groups:
        score += 3.0
        reasons.append(f"group={group}")
    if group in spec.discouraged_groups:
        score -= 1.5
        reasons.append(f"discourage={group}")
    if family in spec.preferred_families:
        score += 4.0
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
        raise ValueError(f"Unknown MOT recommendation profile: {profile!r}. Supported: {supported}")

    variant_name = str(variant).strip().lower()
    if variant_name not in {"tiny", "small", "base"}:
        raise ValueError(f"Unknown variant: {variant!r}. Supported: tiny, small, base")
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    from dlhub.vision.mot_zoo import list_local_arches

    arches_set = set(list_local_arches())
    candidates: list[Recommendation] = []
    for e in entries():
        arch_id = f"mot2d:{e.family}_{variant_name}"
        if arch_id not in arches_set:
            continue
        score, reason = _score_family(spec, family=e.family, group=e.group, year=e.year)
        candidates.append(
            Recommendation(
                profile=profile_key,
                arch_id=arch_id,
                family=e.family,
                group=e.group,
                year=e.year,
                score=score,
                reason=reason,
            )
        )

    candidates.sort(
        key=lambda x: (
            -float(x.score),
            9999 if x.year is None else -int(x.year),
            x.family,
        )
    )
    return candidates[: int(top_k)]


__all__ = ["ProfileSpec", "Recommendation", "list_profiles", "recommend_arches"]
