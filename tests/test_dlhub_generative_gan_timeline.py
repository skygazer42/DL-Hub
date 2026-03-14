import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_gan_timeline_metadata_covers_24_families() -> None:
    from dlhub.generative.gan._timeline import by_family, entries

    timeline = entries()
    assert len(timeline) >= 24

    groups = {entry.group for entry in timeline}
    assert groups == {
        "vanilla_adversarial",
        "conditional_gan",
        "image_translation",
        "high_fidelity",
    }

    mapping = by_family()
    assert mapping["dcgan"].year == 2015
    assert mapping["wgangp"].group == "vanilla_adversarial"
    assert mapping["hingegan"].group == "vanilla_adversarial"
    assert mapping["cgan"].group == "conditional_gan"
    assert mapping["projection_gan"].group == "conditional_gan"
    assert mapping["pix2pix"].group == "image_translation"
    assert mapping["cutgan"].group == "image_translation"
    assert mapping["stylegan2"].group == "high_fidelity"
    assert mapping["stylegan3"].group == "high_fidelity"


def test_gan_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/gan_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "GAN timeline" in proc.stdout
    assert "total_families=" in proc.stdout
    assert "\n2015\n" in proc.stdout
    assert "dcgan [vanilla_adversarial]" in proc.stdout
    assert "cgan [conditional_gan]" in proc.stdout
    assert "stylegan2 [high_fidelity]" in proc.stdout
