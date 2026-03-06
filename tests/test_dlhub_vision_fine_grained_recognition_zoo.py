from __future__ import annotations

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
    if isinstance(x, (list, tuple)):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in fgvc zoo smoke: {type(x)!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_fgvc_zoo_lists_120_plus_arches() -> None:
    from dlhub.vision.fine_grained_recognition_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 120
    assert "dlfgvc:bilinear_cnn_tiny" in arches
    assert "dlfgvc:racnn_tiny" in arches
    assert "dlfgvc:ws_dan_tiny" in arches
    assert "dlfgvc:transfg_tiny" in arches
    assert "dlfgvc:metaformer_fgvc_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.vision.fine_grained_recognition_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_fgvc_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.vision.fine_grained_recognition_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=5, image_size=64, width_mult=0.5, dropout=0.0)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert "logits" in out
    assert tuple(out["logits"].shape) == (2, 5)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("arch_id", ["dlfgvc:bilinear_cnn_small", "dlfgvc:ws_dan_base", "dlfgvc:transfg_small"])
def test_fgvc_zoo_builds_non_tiny_variants(arch_id: str) -> None:
    from dlhub.vision.fine_grained_recognition_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=5, image_size=64, width_mult=0.5, dropout=0.0)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert "logits" in out
    assert tuple(out["logits"].shape) == (1, 5)


def test_fgvc_zoo_script_list_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/fine_grained_recognition_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Fine-grained recognition local zoo" in list_proc.stdout
    assert "total_arches=120" in list_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/fine_grained_recognition_zoo.py", "--smoke", "dlfgvc:transfg_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: dlfgvc:transfg_tiny" in smoke_proc.stdout
