import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in panoptic segmentation zoo smoke: {type(x)!r}")


def test_panoptic_segmentation_zoo_lists_120_plus_arches() -> None:
    from dlhub.vision.panoptic_segmentation_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 120

    assert "dlpan:panoptic_fpn_tiny" in arches
    assert "dlpan:mask2former_panoptic_tiny" in arches
    assert "dlpan:panoptic_deeplab_tiny" in arches
    assert "dlpan:transunet_panoptic_tiny" in arches
    assert "dlpan:rtdetr_panoptic_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.vision.panoptic_segmentation_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_panoptic_segmentation_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.vision.panoptic_segmentation_zoo import build_local_model

    model = build_local_model(
        arch_id, in_channels=3, num_thing_classes=3, num_stuff_classes=2, width_mult=0.5
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize(
    "arch_id",
    [
        "dlpan:panoptic_fpn_small",
        "dlpan:mask2former_panoptic_base",
        "dlpan:transunet_panoptic_small",
    ],
)
def test_panoptic_segmentation_zoo_builds_non_tiny_variants(arch_id: str) -> None:
    from dlhub.vision.panoptic_segmentation_zoo import build_local_model

    model = build_local_model(
        arch_id, in_channels=3, num_thing_classes=3, num_stuff_classes=2, width_mult=0.5
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)


def test_panoptic_segmentation_zoo_script_list_and_smoke() -> None:
    from dlhub.vision.panoptic_segmentation_zoo import list_local_arches

    list_proc = subprocess.run(
        [sys.executable, "scripts/panoptic_segmentation_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Panoptic segmentation local zoo" in list_proc.stdout
    assert f"total_arches={len(list_local_arches())}" in list_proc.stdout
    assert list_proc.stdout.count("dlpan:") <= 8

    smoke_proc = subprocess.run(
        [
            sys.executable,
            "scripts/panoptic_segmentation_zoo.py",
            "--smoke",
            "dlpan:mask2former_panoptic_tiny",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: dlpan:mask2former_panoptic_tiny" in smoke_proc.stdout
