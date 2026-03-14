import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_vlm_timeline_metadata_covers_12_families() -> None:
    from dlhub.multimodal.vlm._timeline import by_family, entries

    timeline = entries()
    assert len(timeline) >= 12

    groups = {entry.group for entry in timeline}
    assert groups == {
        "single_stream",
        "dual_encoder",
        "fusion_encoder_decoder",
        "multimodal_llm",
    }

    mapping = by_family()
    assert mapping["clip"].year == 2021
    assert mapping["blip"].year == 2022
    assert mapping["llava"].year == 2023
    assert mapping["vilt"].group == "single_stream"
    assert mapping["blip2"].group == "multimodal_llm"


def test_vlm_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/vlm_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "VLM timeline" in proc.stdout
    assert "total_families=" in proc.stdout
    assert "\n2021\n" in proc.stdout
    assert "\n2022\n" in proc.stdout
    assert "\n2023\n" in proc.stdout
    assert "clip [dual_encoder]" in proc.stdout
    assert "blip [fusion_encoder_decoder]" in proc.stdout
    assert "llava [multimodal_llm]" in proc.stdout
