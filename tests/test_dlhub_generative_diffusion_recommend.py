import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_diffusion_recommend_profiles_include_core_scenarios() -> None:
    from dlhub.generative.diffusion._recommend import list_profiles

    keys = {p.key for p in list_profiles()}
    assert "balanced" in keys
    assert "lightweight" in keys
    assert "fidelity" in keys
    assert "latent" in keys
    assert "fast_sampling" in keys


def test_diffusion_recommend_fidelity_returns_high_priority_bias() -> None:
    from dlhub.generative.diffusion._recommend import recommend_arches
    from dlhub.generative.diffusion_zoo import list_local_arches

    recs = recommend_arches("fidelity", variant="tiny", top_k=4)
    arches = set(list_local_arches())
    assert len(recs) == 4
    assert all(r.arch_id in arches for r in recs)
    assert any(r.family == "edm" for r in recs)
    assert all(r.group in {"score_based", "latent_diffusion", "flow_matching"} for r in recs)


def test_diffusion_zoo_script_recommend_and_profiles() -> None:
    profiles_proc = subprocess.run(
        [sys.executable, "scripts/diffusion_zoo.py", "--list-profiles"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert profiles_proc.returncode == 0
    assert "Diffusion recommendation profiles" in profiles_proc.stdout
    assert "fidelity" in profiles_proc.stdout

    rec_proc = subprocess.run(
        [
            sys.executable,
            "scripts/diffusion_zoo.py",
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
    assert "Diffusion recommendations" in rec_proc.stdout
    assert "profile=fidelity" in rec_proc.stdout
    assert "diff:" in rec_proc.stdout
