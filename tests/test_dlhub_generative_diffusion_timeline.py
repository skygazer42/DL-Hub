import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_diffusion_timeline_metadata_covers_12_families() -> None:
    from dlhub.generative.diffusion._timeline import by_family, entries

    timeline = entries()
    assert len(timeline) >= 12

    groups = {entry.group for entry in timeline}
    assert groups == {
        "pixel_diffusion",
        "score_based",
        "latent_diffusion",
        "flow_matching",
    }

    mapping = by_family()
    assert mapping["ddpm"].year == 2020
    assert mapping["ddim"].group == "pixel_diffusion"
    assert mapping["score_sde"].group == "score_based"
    assert mapping["latent_diffusion"].group == "latent_diffusion"
    assert mapping["flow_matching"].group == "flow_matching"


def test_diffusion_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/diffusion_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Diffusion timeline" in proc.stdout
    assert "total_families=" in proc.stdout
    assert "\n2020\n" in proc.stdout
    assert "ddpm [pixel_diffusion]" in proc.stdout
    assert "score_sde [score_based]" in proc.stdout
    assert "latent_diffusion [latent_diffusion]" in proc.stdout
