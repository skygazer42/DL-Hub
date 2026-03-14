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
        summary="Mix Kalman, BEV, and segmentation based 3D trackers.",
        preferred_groups=("kalman_association", "bev_tracking", "segmentation_tracking"),
        preferred_families=(
            "ab3dmot",
            "centerpoint_track",
            "bitrack",
            "bevfusion_track",
            "motsf3d",
            "pointtrack3d",
        ),
        modern_bias=0.35,
    ),
    "realtime_lidar": ProfileSpec(
        key="realtime_lidar",
        title="Real-Time LiDAR",
        summary="Prefer lightweight online Kalman and IoU tracking routes.",
        preferred_groups=("kalman_association",),
        preferred_families=(
            "simpletrack",
            "ab3dmot",
            "ocsort3d",
            "lidar_iou_track",
            "ekf3d",
            "ukf3d",
        ),
        discouraged_groups=("segmentation_tracking",),
        modern_bias=0.15,
    ),
    "bev_priority": ProfileSpec(
        key="bev_priority",
        title="BEV Priority",
        summary="Prioritize bird's-eye-view tracking families for AV stacks.",
        preferred_groups=("bev_tracking",),
        preferred_families=(
            "centerpoint_track",
            "bitrack",
            "bevsort",
            "bevfusion_track",
            "voxeltrack",
            "centertrack3d",
            "pillartrack",
            "transcenter3d",
            "centerbev_track",
            "motionbev_track",
            "querybev_track",
            "sparsebev_track",
            "mapbev_track",
            "hdmap_bev_track",
            "lanebev_track",
            "occupancy_bev_track",
            "temporalbev_track",
            "velocitybev_track",
            "scenebev_track",
            "multimodal_bev_track",
            "anchorfree_bev_track",
            "transformbev_track",
            "streambev_track",
            "bevformer_track",
            "bevnext_track",
            "depthbev_track",
            "graphbev_track",
            "memorybev_track",
            "radarbev_track",
            "stereo_bev_track",
            "trajectorybev_track",
            "uncertaintybev_track",
            "worldbev_track",
            "mapprior_bev_track",
            "vectorbev_track",
            "crossview_bev_track",
            "liftbev_track",
            "occupancyflow_bev_track",
            "sparseformer_bev_track",
            "eventbev_track",
            "planningbev_track",
            "topologybev_track",
            "geobev_track",
            "cambev_track",
            "lidarbev_track",
            "radarfusion_bev_track",
            "maplane_bev_track",
            "scenegraph_bev_track",
            "interactivebev_track",
            "predictivebev_track",
            "globalbev_track",
            "hyperbev_track",
            "robustbev_track",
            "lowlatency_bev_track",
            "tinybev_track",
            "quantbev_track",
            "edgebev_track",
            "compressedbev_track",
            "distillbev_track",
            "mobilebev_track",
            "fastmap_bev_track",
            "agilebev_track",
            "streamlite_bev_track",
            "ultrafast_bev_track",
            "realtime_bev_track",
            "nanobev_track",
            "microbev_track",
            "econobev_track",
            "slimbev_track",
            "swiftbev_track",
            "powerbev_track",
            "budgetbev_track",
            "turbo_bev_track",
            "sensorlite_bev_track",
            "ondevice_bev_track",
            "lowpower_bev_track",
            "cachebev_track",
            "instantbev_track",
            "rapidbev_track",
            "frugalbev_track",
            "compactbev_track",
            "sparselite_bev_track",
            "latencyguard_bev_track",
            "ultralite_bev_track",
            "minipower_bev_track",
            "featherbev_track",
            "scoutbev_track",
            "zipbev_track",
            "thriftbev_track",
            "flashbev_track",
            "zipstream_bev_track",
            "quickmap_bev_track",
            "nanoedge_bev_track",
            "pulsebev_track",
            "briskbev_track",
            "sprintbev_track",
            "leanbev_track",
            "rangerbev_track",
            "depotbev_track",
            "meshbev_track",
            "relaybev_track",
            "nimblebev_track",
            "steadyedge_bev_track",
        ),
        modern_bias=0.5,
    ),
    "segmentation_first": ProfileSpec(
        key="segmentation_first",
        title="Segmentation-Tracking",
        summary="Prioritize segmentation-guided 3D tracking pipelines.",
        preferred_groups=("segmentation_tracking",),
        preferred_families=(
            "motsf3d",
            "pointtrack3d",
            "masktrack3d",
            "segtrack3d",
            "panoptictrack3d",
            "instanceflow3d",
            "trackletseg3d",
        ),
        modern_bias=0.4,
    ),
    "long_horizon": ProfileSpec(
        key="long_horizon",
        title="Long-Horizon Stability",
        summary="Prefer trackers that are stable on long temporal windows.",
        preferred_groups=("kalman_association", "bev_tracking"),
        preferred_families=(
            "imm_kalman",
            "ukf3d",
            "ekf3d",
            "ma3dmot",
            "deepsort3d",
            "transcenter3d",
            "bevfusion_track",
        ),
        modern_bias=0.3,
    ),
}


def list_profiles() -> list[ProfileSpec]:
    return [*(_PROFILES[k] for k in sorted(_PROFILES))]


def _family_year_bonus(year: int | None, *, modern_bias: float) -> float:
    if year is None:
        return 0.0
    span = max(0.0, min(1.0, (float(year) - 2018.0) / 8.0))
    return float(modern_bias) * span


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
        raise ValueError(f"Unknown Tracking3D recommendation profile: {profile!r}. Supported: {supported}")

    variant_name = str(variant).strip().lower()
    if variant_name not in {"tiny", "small", "base"}:
        raise ValueError(f"Unknown variant: {variant!r}. Supported: tiny, small, base")
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    from dlhub.pointcloud.tracking3d_zoo import list_local_arches

    arches_set = set(list_local_arches())
    candidates: list[Recommendation] = []
    for e in entries():
        arch_id = f"pctrk3d:{e.family}_{variant_name}"
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
