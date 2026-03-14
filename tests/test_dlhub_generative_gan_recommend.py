import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_gan_recommend_profiles_include_core_scenarios() -> None:
    from dlhub.generative.gan._recommend import list_profiles

    keys = {p.key for p in list_profiles()}
    assert "balanced" in keys
    assert "lightweight" in keys
    assert "fidelity" in keys
    assert "conditional" in keys
    assert "stable_training" in keys


def test_gan_recommend_fidelity_returns_high_fidelity_bias() -> None:
    from dlhub.generative.gan._recommend import recommend_arches
    from dlhub.generative.gan_zoo import list_local_arches

    recs = recommend_arches("fidelity", variant="tiny", top_k=4)
    arches = set(list_local_arches())
    assert len(recs) == 4
    assert all(r.arch_id in arches for r in recs)
    assert all(r.group == "high_fidelity" for r in recs)
    assert any(r.family == "stylegan2" for r in recs)


def test_gan_zoo_script_recommend_and_profiles() -> None:
    profiles_proc = subprocess.run(
        [sys.executable, "scripts/gan_zoo.py", "--list-profiles"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert profiles_proc.returncode == 0
    assert "GAN recommendation profiles" in profiles_proc.stdout
    assert "fidelity" in profiles_proc.stdout

    rec_proc = subprocess.run(
        [
            sys.executable,
            "scripts/gan_zoo.py",
            "--recommend",
            "fidelity",
            "--variant",
            "tiny",
            "--top-k",
            "4",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert rec_proc.returncode == 0
    assert "GAN recommendations" in rec_proc.stdout
    assert "profile=fidelity" in rec_proc.stdout
    assert "gan:" in rec_proc.stdout

