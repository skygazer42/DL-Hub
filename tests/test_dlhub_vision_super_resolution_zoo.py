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
    raise TypeError(f"Unsupported output type in super-resolution zoo smoke: {type(x)!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_super_resolution_zoo_lists_families() -> None:
    from dlhub.vision.super_resolution_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 18
    assert "sr:srcnn_tiny" in arches
    assert "sr:fsrcnn_small" in arches
    assert "sr:edsr_sr_base" in arches
    assert "sr:rcan_sr_tiny" in arches
    assert "sr:rdn_sr_small" in arches
    assert "sr:swinir_sr_tiny" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "sr:srcnn_tiny",
        "sr:fsrcnn_tiny",
        "sr:edsr_sr_tiny",
        "sr:rcan_sr_tiny",
        "sr:rdn_sr_tiny",
        "sr:swinir_sr_tiny",
    ],
)
def test_super_resolution_zoo_build_and_forward_smoke(arch_id: str) -> None:
    from dlhub.vision.super_resolution_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        upscale_factor=2,
        image_size=16,
        width_mult=0.5,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    assert isinstance(out, dict)
    assert "sr" in out
    assert tuple(out["sr"].shape) == (2, 3, 32, 32)

    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()


def test_super_resolution_zoo_script_list_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/super_resolution_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Super-resolution local zoo" in list_proc.stdout
    assert "total_arches=" in list_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/super_resolution_zoo.py", "--smoke", "sr:srcnn_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: sr:srcnn_tiny" in smoke_proc.stdout
