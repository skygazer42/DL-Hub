"""Diffusion timeline metadata (best effort, for docs and CLI)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEntry:
    year: int | None
    family: str
    method: str
    group: str
    reference: str | None = None


_ENTRIES: list[TimelineEntry] = [
    TimelineEntry(
        2020, "ddpm", "DDPM (denoising diffusion probabilistic model)", "pixel_diffusion"
    ),
    TimelineEntry(
        2020, "ddim", "DDIM (deterministic implicit diffusion sampling)", "pixel_diffusion"
    ),
    TimelineEntry(2021, "iddpm", "Improved DDPM (learned variance diffusion)", "pixel_diffusion"),
    TimelineEntry(2021, "score_sde", "Score SDE (score-based SDE framework)", "score_based"),
    TimelineEntry(2020, "ncsnpp", "NCSN++ (improved score network family)", "score_based"),
    TimelineEntry(2022, "edm", "EDM (elucidated diffusion models)", "score_based"),
    TimelineEntry(
        2022,
        "latent_diffusion",
        "Latent Diffusion (autoencoded latent denoising)",
        "latent_diffusion",
    ),
    TimelineEntry(
        2022,
        "stable_diffusion",
        "Stable Diffusion (text-aligned latent diffusion)",
        "latent_diffusion",
    ),
    TimelineEntry(
        2023,
        "consistency_model",
        "Consistency Model (few-step consistency distillation)",
        "latent_diffusion",
    ),
    TimelineEntry(
        2023, "flow_matching", "Flow Matching (vector field transport objective)", "flow_matching"
    ),
    TimelineEntry(
        2023, "rectified_flow", "Rectified Flow (straightened flow trajectories)", "flow_matching"
    ),
    TimelineEntry(
        2023,
        "conditional_flow_matching",
        "Conditional Flow Matching (conditional vector field training)",
        "flow_matching",
    ),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {entry.family: entry for entry in _ENTRIES}
