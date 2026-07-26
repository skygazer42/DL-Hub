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
        title="Balanced GAN Starter",
        summary="Mix baseline, conditional, translation, and fidelity families.",
        preferred_groups=(
            "vanilla_adversarial",
            "conditional_gan",
            "image_translation",
            "high_fidelity",
        ),
        preferred_families=(
            "dcgan",
            "wgangp",
            "hingegan",
            "cgan",
            "infogan",
            "pix2pix",
            "cutgan",
            "stylegan2",
            "stylegan3",
        ),
        modern_bias=0.25,
    ),
    "lightweight": ProfileSpec(
        key="lightweight",
        title="Lightweight GAN",
        summary="Prefer simple and stable adversarial baselines.",
        preferred_groups=("vanilla_adversarial",),
        preferred_families=("dcgan", "lsgan", "wgangp", "hingegan", "dragan"),
        discouraged_groups=("high_fidelity",),
        modern_bias=0.1,
    ),
    "fidelity": ProfileSpec(
        key="fidelity",
        title="High Fidelity",
        summary="Prefer style and attention heavy high-fidelity families.",
        preferred_groups=("high_fidelity",),
        preferred_families=(
            "stylegan3",
            "stylegan2",
            "stylegan",
            "biggan",
            "sagan",
            "progan",
            "transgan",
        ),
        modern_bias=0.35,
    ),
    "conditional": ProfileSpec(
        key="conditional",
        title="Conditional and Translation",
        summary="Prefer label-conditioned and image-translation GAN families.",
        preferred_groups=("conditional_gan", "image_translation"),
        preferred_families=(
            "cgan",
            "acgan",
            "projection_gan",
            "infogan",
            "pix2pix",
            "cyclegan",
            "cutgan",
        ),
        modern_bias=0.2,
    ),
    "stable_training": ProfileSpec(
        key="stable_training",
        title="Stable Training",
        summary="Prefer smoother objectives and stable discriminator dynamics.",
        preferred_groups=("vanilla_adversarial",),
        preferred_families=("wgangp", "wgan", "lsgan", "hingegan", "dragan", "sagan"),
        modern_bias=0.15,
    ),
}


def list_profiles() -> list[ProfileSpec]:
    return [*(_PROFILES[k] for k in sorted(_PROFILES))]


def _family_year_bonus(year: int | None, *, modern_bias: float) -> float:
    if year is None:
        return 0.0
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
        raise ValueError(f"Unknown GAN recommendation profile: {profile!r}. Supported: {supported}")

    variant_name = str(variant).strip().lower()
    if variant_name not in {"tiny", "small", "base"}:
        raise ValueError(f"Unknown variant: {variant!r}. Supported: tiny, small, base")
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    from dlhub.generative.gan_zoo import list_local_arches

    arches_set = set(list_local_arches())
    candidates: list[Recommendation] = []
    for e in entries():
        arch_id = f"gan:{e.family}_{variant_name}"
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
