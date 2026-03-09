"""3D tracking timeline metadata (best-effort, for docs/CLI)."""

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
    TimelineEntry(2020, "ab3dmot", "AB3DMOT (Kalman + 3D box association)", "kalman_association"),
    TimelineEntry(2021, "centerpoint_track", "CenterPoint-Track (BEV center tracking)", "bev_tracking"),
    TimelineEntry(2022, "simpletrack", "SimpleTrack (lightweight Kalman + affinity)", "kalman_association"),
    TimelineEntry(2023, "bitrack", "BiTrack (bi-directional BEV association)", "bev_tracking"),
    TimelineEntry(2020, "motsf3d", "MOTSF3D (joint 3D segmentation/tracking)", "segmentation_tracking"),
    TimelineEntry(2019, "imm_kalman", "IMM-Kalman (multi-model motion tracking)", "kalman_association"),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}
