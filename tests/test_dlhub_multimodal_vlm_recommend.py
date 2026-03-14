import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_vlm_recommend_profiles_include_core_scenarios() -> None:
    from dlhub.multimodal.vlm._recommend import list_profiles

    keys = {p.key for p in list_profiles()}
    assert "balanced" in keys
    assert "retrieval" in keys
    assert "captioning" in keys
    assert "instruction" in keys
    assert "lightweight" in keys


def test_vlm_recommend_instruction_returns_multimodal_llm_bias() -> None:
    from dlhub.multimodal.vlm._recommend import recommend_arches
    from dlhub.multimodal.vlm_zoo import list_local_arches

    recs = recommend_arches("instruction", variant="tiny", top_k=4)
    arches = set(list_local_arches())
    assert len(recs) == 4
    assert all(rec.arch_id in arches for rec in recs)
    assert recs[0].family == "qwen_vl"
    assert any(rec.family in {"qwen_vl", "cogvlm", "mplug_owl2"} for rec in recs)
    assert all(rec.group in {"multimodal_llm", "fusion_encoder_decoder"} for rec in recs)


def test_vlm_zoo_script_recommend_and_profiles() -> None:
    profiles_proc = subprocess.run(
        [sys.executable, "scripts/vlm_zoo.py", "--list-profiles"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert profiles_proc.returncode == 0
    assert "VLM recommendation profiles" in profiles_proc.stdout
    assert "instruction" in profiles_proc.stdout

    rec_proc = subprocess.run(
        [
            sys.executable,
            "scripts/vlm_zoo.py",
            "--recommend",
            "instruction",
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
    assert "VLM recommendations" in rec_proc.stdout
    assert "profile=instruction" in rec_proc.stdout
    assert "vlm:" in rec_proc.stdout
