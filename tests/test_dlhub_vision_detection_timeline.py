import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_detection_timeline_metadata_covers_archive() -> None:
    from dlhub.vision.detection._timeline import by_family, entries

    timeline = entries()
    assert len(timeline) >= 101

    groups = {entry.group for entry in timeline}
    assert groups == {
        "single_stage",
        "two_stage",
        "keypoint_anchor_free",
        "transformer_query",
        "open_vocabulary_multimodal",
    }

    mapping = by_family()
    assert mapping["rcnn"].year == 2014
    assert mapping["rcnn"].group == "two_stage"
    assert mapping["yolov4"].group == "single_stage"
    assert mapping["centernet2"].group == "keypoint_anchor_free"
    assert mapping["co_detr"].group == "transformer_query"
    assert mapping["glip"].group == "open_vocabulary_multimodal"
    assert mapping["yolo11"].year == 2024
    assert mapping["d_fine"].group == "transformer_query"
    assert mapping["ovlw_detr"].group == "open_vocabulary_multimodal"
    assert mapping["rtgen"].year == 2025
    assert mapping["sa_detr"].group == "transformer_query"
    assert mapping["yolo26"].year == 2026
    assert mapping["yolo_world"].year >= 2024


def test_detection_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/detection_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Detection timeline" in proc.stdout
    assert "total_families=" in proc.stdout
    assert "\n2014\n" in proc.stdout
    assert "\n2026\n" in proc.stdout
    assert "rcnn [two_stage]" in proc.stdout
    assert "yolov4 [single_stage]" in proc.stdout
    assert "co_detr [transformer_query]" in proc.stdout
    assert "glip [open_vocabulary_multimodal]" in proc.stdout
    assert "yolo26 [single_stage]" in proc.stdout
