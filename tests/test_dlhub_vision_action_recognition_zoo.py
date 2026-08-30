import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_action_recognition_zoo_lists_20_plus_arches() -> None:
    from dlhub.vision.action_recognition_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 20
    assert arches == sorted(arches)

    assert "dlactv:c3d_tiny" in arches
    assert "dlactv:tsn_tiny" in arches
    assert "dlactv:tsm_tiny" in arches
    assert "dlactv:slowfast_tiny" in arches
    assert "dlactv:timesformer_tiny" in arches

    assert "dlacts:stgcn_tiny" in arches
    assert "dlacts:agcn_tiny" in arches
    assert "dlacts:ctr_gcn_tiny" in arches
    assert "dlacts:poseformer_tiny" in arches
    assert "dlacts:shift_gcn_tiny" in arches
    assert "dlacts:motionbert_tiny" in arches

    assert "dlactv:two_stream_tiny" in arches
    assert "dlactv:videomae_tiny" in arches
    assert "dlactv:videomamba_tiny" in arches
    assert "dlactv:videornn_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.vision.action_recognition_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_action_recognition_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.vision.action_recognition_zoo import build_local_model

    if arch_id.startswith("dlacts:"):
        model = build_local_model(
            arch_id,
            in_channels=3,
            num_classes=6,
            num_joints=17,
            seq_len=32,
            width_mult=0.5,
            dropout=0.0,
        )
        x = torch.randn(2, 3, 32, 17)
    else:
        model = build_local_model(
            arch_id,
            in_channels=3,
            num_classes=6,
            image_size=64,
            frames=8,
            width_mult=0.5,
            dropout=0.0,
        )
        x = torch.randn(2, 3, 8, 64, 64)

    out = model(x)
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (2, 6)
    loss = out.to(torch.float32).mean()
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize(
    "arch_id", ["dlactv:timesformer_small", "dlacts:poseformer_base", "dlacts:ctr_gcn_small"]
)
def test_action_recognition_zoo_builds_non_tiny_variants(arch_id: str) -> None:
    from dlhub.vision.action_recognition_zoo import build_local_model

    if arch_id.startswith("dlacts:"):
        model = build_local_model(
            arch_id,
            in_channels=3,
            num_classes=6,
            num_joints=17,
            seq_len=32,
            width_mult=0.5,
            dropout=0.0,
        )
        x = torch.randn(1, 3, 32, 17)
    else:
        model = build_local_model(
            arch_id,
            in_channels=3,
            num_classes=6,
            image_size=64,
            frames=8,
            width_mult=0.5,
            dropout=0.0,
        )
        x = torch.randn(1, 3, 8, 64, 64)

    out = model(x)
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (1, 6)


def test_action_recognition_zoo_script_list_and_smoke_and_timeline() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/action_recognition_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Action recognition local zoo" in list_proc.stdout
    assert "total_arches=" in list_proc.stdout
    assert "dlactv=" in list_proc.stdout
    assert "dlacts=" in list_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/action_recognition_zoo.py", "--smoke", "dlactv:c3d_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: dlactv:c3d_tiny" in smoke_proc.stdout

    smoke_proc2 = subprocess.run(
        [sys.executable, "scripts/action_recognition_zoo.py", "--smoke", "dlacts:stgcn_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc2.returncode == 0
    assert "smoke: dlacts:stgcn_tiny" in smoke_proc2.stdout

    timeline_proc = subprocess.run(
        [sys.executable, "scripts/action_recognition_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert timeline_proc.returncode == 0
    assert "Action recognition timeline" in timeline_proc.stdout
    assert "\n2015\n" in timeline_proc.stdout
    assert "c3d" in timeline_proc.stdout
    assert "stgcn" in timeline_proc.stdout
