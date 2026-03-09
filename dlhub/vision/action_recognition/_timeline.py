
"""Action recognition timeline metadata (best-effort, for docs/CLI).

Notes:
- Years are based on representative papers or the earliest commonly-cited version.
- Models in this repo are "toy interpretations" of an idea, not strict reproductions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEntry:
    year: int | None
    family: str
    method: str
    group: str  # "video" | "skeleton"
    reference: str | None = None


# Source of truth for `scripts/action_recognition_zoo.py --timeline` and README tables.
_ENTRIES: list[TimelineEntry] = [
    # --- video action recognition
    TimelineEntry(2014, "two_stream", "Two-Stream CNN (RGB + motion stream, toy)", "video"),
    TimelineEntry(2015, "c3d", "C3D (3D CNN baseline)", "video"),
    TimelineEntry(2016, "tsn", "TSN (segment sampling + consensus)", "video"),
    TimelineEntry(2017, "i3d", "I3D (inflated 3D conv, toy)", "video"),
    TimelineEntry(2018, "r2plus1d", "R(2+1)D (factorized 3D conv, toy)", "video"),
    TimelineEntry(2018, "non_local", "Non-local block (space-time self-attention, toy)", "video"),
    TimelineEntry(2019, "tsm", "TSM (temporal shift module)", "video"),
    TimelineEntry(2019, "slowfast", "SlowFast (dual-pathway)", "video"),
    TimelineEntry(2020, "x3d", "X3D (efficient 3D conv, toy)", "video"),
    TimelineEntry(2021, "timesformer", "TimeSformer (space-time attention)", "video"),
    TimelineEntry(2021, "vivit", "ViViT (factorized video transformer, toy)", "video"),
    TimelineEntry(2022, "videomae", "VideoMAE (tubelet ViT, toy)", "video"),
    TimelineEntry(2024, "videomamba", "VideoMamba (SSM/Mamba-style mixer, toy)", "video"),
    TimelineEntry(2025, "videornn", "VideoRNN (CNN+GRU, efficient temporal modeling, toy)", "video"),
    # --- skeleton-based action recognition
    TimelineEntry(2018, "stgcn", "ST-GCN (spatio-temporal graph conv)", "skeleton"),
    TimelineEntry(2019, "agcn", "2S-AGCN (adaptive graph conv, toy)", "skeleton"),
    TimelineEntry(2020, "shift_gcn", "Shift-GCN (shift operator on joints/time, toy)", "skeleton"),
    TimelineEntry(2020, "ms_g3d", "MS-G3D (multi-hop graph conv, toy)", "skeleton"),
    TimelineEntry(2021, "ctr_gcn", "CTR-GCN (dynamic topology refinement, toy)", "skeleton"),
    TimelineEntry(2021, "poseformer", "PoseFormer (transformer over joints/time, toy)", "skeleton"),
    TimelineEntry(2021, "sttr", "ST-Transformer (factorized spatial+temporal attention, toy)", "skeleton"),
    TimelineEntry(2022, "motionbert", "MotionBERT (masked motion modeling, toy)", "skeleton"),
]


def entries() -> list[TimelineEntry]:
    return list(_ENTRIES)


def by_family() -> dict[str, TimelineEntry]:
    return {e.family: e for e in _ENTRIES}
