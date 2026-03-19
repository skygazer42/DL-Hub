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
    raise TypeError(f"Unsupported output type in co-segmentation zoo smoke: {type(x)!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_co_segmentation_zoo_lists_families() -> None:
    from dlhub.vision.co_segmentation_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 18
    assert "coseg:siamese_coseg_tiny" in arches
    assert "coseg:cosal_uformer_small" in arches
    assert "coseg:group_proto_net_base" in arches
    assert "coseg:co_attention_fpn_tiny" in arches
    assert "coseg:transformer_coseg_small" in arches
    assert "coseg:consensus_refiner_base" in arches


def test_co_segmentation_common_flatten_roundtrip() -> None:
    from dlhub.vision.co_segmentation._common import flatten_group, unflatten_group

    x = torch.randn(2, 3, 4, 8, 8)
    flat = flatten_group(x)
    y = unflatten_group(flat, batch=2, set_size=3)
    assert tuple(flat.shape) == (6, 4, 8, 8)
    assert torch.allclose(x, y)


@pytest.mark.parametrize(
    "arch_id",
    [
        "coseg:siamese_coseg_tiny",
        "coseg:cosal_uformer_tiny",
        "coseg:group_proto_net_tiny",
        "coseg:co_attention_fpn_tiny",
        "coseg:transformer_coseg_tiny",
        "coseg:consensus_refiner_tiny",
    ],
)
def test_co_segmentation_zoo_build_and_forward_smoke(arch_id: str) -> None:
    from dlhub.vision.co_segmentation_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        num_classes=2,
        image_size=64,
        set_size=3,
        width_mult=0.5,
        dropout=0.0,
    )

    images = torch.randn(2, 3, 3, 64, 64)
    out = model(images)
    assert isinstance(out, dict)
    assert "logits" in out
    assert "masks" in out
    assert tuple(out["logits"].shape) == (2, 3, 2, 64, 64)
    assert tuple(out["masks"].shape) == (2, 3, 64, 64)

    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()


def test_co_segmentation_zoo_script_list_search_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/co_segmentation_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Co-segmentation local zoo" in list_proc.stdout
    assert "total_arches=" in list_proc.stdout

    search_proc = subprocess.run(
        [sys.executable, "scripts/co_segmentation_zoo.py", "--search", "transformer_coseg", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert search_proc.returncode == 0
    assert "coseg:transformer_coseg_tiny" in search_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/co_segmentation_zoo.py", "--smoke", "coseg:siamese_coseg_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: coseg:siamese_coseg_tiny" in smoke_proc.stdout
