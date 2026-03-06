import pytest
import subprocess
import sys
from pathlib import Path


torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, (list, tuple)):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in instance segmentation zoo smoke: {type(x)!r}")


def test_instance_segmentation_zoo_lists_120_plus_arches() -> None:
    from dlhub.vision.instance_segmentation_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 120

    assert "dlinst:yolact_tiny" in arches
    assert "dlinst:mask_rcnn_tiny" in arches
    assert "dlinst:deepmask_tiny" in arches
    assert "dlinst:sharpmask_tiny" in arches
    assert "dlinst:mnc_tiny" in arches
    assert "dlinst:instancefcn_tiny" in arches
    assert "dlinst:sipmask_tiny" in arches
    assert "dlinst:mask_dino_tiny" in arches
    assert "dlinst:deepsnake_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.vision.instance_segmentation_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_instance_segmentation_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.vision.instance_segmentation_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=2, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("arch_id", ["dlinst:deepmask_small", "dlinst:mask_dino_base", "dlinst:rtmdet_ins_small"])
def test_instance_segmentation_zoo_builds_non_tiny_variants(arch_id: str) -> None:
    from dlhub.vision.instance_segmentation_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=2, width_mult=0.5)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)


def test_instance_segmentation_zoo_script_list_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/instance_segmentation_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Instance segmentation local zoo" in list_proc.stdout
    assert "total_arches=120" in list_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/instance_segmentation_zoo.py", "--smoke", "dlinst:mask_dino_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: dlinst:mask_dino_tiny" in smoke_proc.stdout
