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
    raise TypeError(f"Unsupported output type in face parsing zoo smoke: {type(x)!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_face_parsing_zoo_lists_families() -> None:
    from dlhub.vision.face_parsing_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 36
    assert "fparse:roi_tanh_warp_tiny" in arches
    assert "fparse:dml_csr_small" in arches
    assert "fparse:fp_liif_base" in arches
    assert "fparse:stn_icnn_tiny" in arches
    assert "fparse:segface_small" in arches
    assert "fparse:facexformer_parse_base" in arches
    assert "fparse:occlusion_tanh_tiny" in arches
    assert "fparse:mask_fpan_small" in arches
    assert "fparse:farl_parse_base" in arches
    assert "fparse:eagrnet_tiny" in arches
    assert "fparse:agrnet_small" in arches
    assert "fparse:ehanet_base" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "fparse:roi_tanh_warp_tiny",
        "fparse:dml_csr_tiny",
        "fparse:fp_liif_tiny",
        "fparse:stn_icnn_tiny",
        "fparse:segface_tiny",
        "fparse:facexformer_parse_tiny",
        "fparse:occlusion_tanh_tiny",
        "fparse:mask_fpan_tiny",
        "fparse:farl_parse_tiny",
        "fparse:eagrnet_tiny",
        "fparse:agrnet_tiny",
        "fparse:ehanet_tiny",
    ],
)
def test_face_parsing_zoo_build_and_forward_smoke(arch_id: str) -> None:
    from dlhub.vision.face_parsing_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        num_classes=11,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
    )

    image = torch.randn(2, 3, 64, 64)
    out = model(image)
    assert isinstance(out, dict)
    assert "logits" in out
    assert "parsing_map" in out
    assert tuple(out["logits"].shape) == (2, 11, 64, 64)
    assert tuple(out["parsing_map"].shape) == (2, 64, 64)

    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()


def test_face_parsing_zoo_script_list_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/face_parsing_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Face parsing local zoo" in list_proc.stdout
    assert "total_arches=" in list_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/face_parsing_zoo.py", "--smoke", "fparse:roi_tanh_warp_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: fparse:roi_tanh_warp_tiny" in smoke_proc.stdout
