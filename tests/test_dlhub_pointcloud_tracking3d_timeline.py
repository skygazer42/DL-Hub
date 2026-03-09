import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_tracking3d_timeline_metadata_covers_first_batch() -> None:
    from dlhub.pointcloud.tracking3d._timeline import by_family, entries

    timeline = entries()
    assert len(timeline) >= 6

    groups = {entry.group for entry in timeline}
    assert groups == {
        "kalman_association",
        "bev_tracking",
        "segmentation_tracking",
    }

    mapping = by_family()
    assert mapping["ab3dmot"].year == 2020
    assert mapping["centerpoint_track"].group == "bev_tracking"
    assert mapping["simpletrack"].group == "kalman_association"
    assert mapping["bitrack"].group == "bev_tracking"
    assert mapping["motsf3d"].group == "segmentation_tracking"
    assert mapping["imm_kalman"].group == "kalman_association"


def test_tracking3d_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/tracking3d_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Tracking3D timeline" in proc.stdout
    assert "total_families=" in proc.stdout
    assert "\n2020\n" in proc.stdout
    assert "ab3dmot [kalman_association]" in proc.stdout
    assert "centerpoint_track [bev_tracking]" in proc.stdout
    assert "motsf3d [segmentation_tracking]" in proc.stdout
