import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_vision_mot_timeline_metadata_covers_80_families() -> None:
    from dlhub.vision.mot._timeline import by_family, entries

    timeline = entries()
    assert len(timeline) >= 80

    groups = {entry.group for entry in timeline}
    assert groups == {
        "online_association",
        "joint_det_embed",
        "query_transformer",
        "global_optimization",
        "probabilistic_filtering",
    }

    mapping = by_family()
    assert mapping["sort"].year == 2016
    assert mapping["deepsort"].group == "online_association"
    assert mapping["fairmot"].group == "joint_det_embed"
    assert mapping["trackformer"].group == "query_transformer"
    assert mapping["network_flow"].group == "global_optimization"
    assert mapping["pmbm_gmphd"].group == "probabilistic_filtering"
    assert mapping["motdt"].group == "online_association"
    assert mapping["motip"].group == "query_transformer"
    assert mapping["uav_sort"].group == "online_association"
    assert mapping["motrv2"].group == "query_transformer"


def test_vision_mot_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/mot_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Vision MOT timeline" in proc.stdout
    assert "total_families=" in proc.stdout
    assert "\n2016\n" in proc.stdout
    assert "sort [online_association]" in proc.stdout
    assert "fairmot [joint_det_embed]" in proc.stdout
    assert "trackformer [query_transformer]" in proc.stdout
