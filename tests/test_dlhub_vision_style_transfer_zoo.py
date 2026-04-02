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
    raise TypeError(f"Unsupported output type in style transfer zoo smoke: {type(x)!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_style_transfer_zoo_lists_style_transfer_families() -> None:
    from dlhub.vision.style_transfer_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 102
    assert "dlst:gatys_tiny" in arches
    assert "dlst:fast_nst_small" in arches
    assert "dlst:adain_base" in arches
    assert "dlst:wct_tiny" in arches
    assert "dlst:pix2pix_tiny" in arches
    assert "dlst:cyclegan_small" in arches
    assert "dlst:cut_base" in arches
    assert "dlst:munit_tiny" in arches

    # Popular additions from ~2018-2022.
    assert "dlst:avatar_net_tiny" in arches
    assert "dlst:sanet_small" in arches
    assert "dlst:ugatit_tiny" in arches
    assert "dlst:starganv2_small" in arches
    assert "dlst:stytr2_tiny" in arches

    # Diffusion-based style transfer (2022+).
    assert "dlst:stylediffusion_tiny" in arches
    assert "dlst:controlnet_small" in arches
    assert "dlst:ip_adapter_tiny" in arches
    assert "dlst:cfg_stylediffusion_tiny" in arches
    assert "dlst:style_aligned_small" in arches

    # More arbitrary style transfer (2021+).
    assert "dlst:adaattn_tiny" in arches

    # Recent diffusion editing / stylization directions (2021+).
    assert "dlst:sdedit_tiny" in arches
    assert "dlst:instantstyle_small" in arches
    assert "dlst:attenst_tiny" in arches

    # Mixed arbitrary / translation family expansion.
    assert "dlst:style_swap_tiny" in arches
    assert "dlst:linear_style_small" in arches
    assert "dlst:photo_wct_base" in arches
    assert "dlst:artflow_tiny" in arches
    assert "dlst:mast_small" in arches
    assert "dlst:cast_tiny" in arches
    assert "dlst:ccpl_base" in arches
    assert "dlst:dualgan_tiny" in arches
    assert "dlst:unit_small" in arches
    assert "dlst:councilgan_tiny" in arches
    assert "dlst:disco_gan_small" in arches
    assert "dlst:whitebox_gan_base" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "dlst:adain_tiny",
        "dlst:wct_tiny",
        "dlst:pix2pix_tiny",
        "dlst:cyclegan_tiny",
        "dlst:munit_tiny",
        "dlst:sanet_tiny",
        "dlst:avatar_net_tiny",
        "dlst:ugatit_tiny",
        "dlst:starganv2_tiny",
        "dlst:stytr2_tiny",
        "dlst:stylediffusion_tiny",
        "dlst:controlnet_tiny",
        "dlst:ip_adapter_tiny",
        "dlst:cfg_stylediffusion_tiny",
        "dlst:style_aligned_tiny",
        "dlst:adaattn_tiny",
        "dlst:sdedit_tiny",
        "dlst:instantstyle_tiny",
        "dlst:attenst_tiny",
        "dlst:style_swap_tiny",
        "dlst:linear_style_tiny",
        "dlst:photo_wct_tiny",
        "dlst:artflow_tiny",
        "dlst:mast_tiny",
        "dlst:cast_tiny",
        "dlst:ccpl_tiny",
        "dlst:dualgan_tiny",
        "dlst:unit_tiny",
        "dlst:councilgan_tiny",
        "dlst:disco_gan_tiny",
        "dlst:whitebox_gan_tiny",
    ],
)
def test_style_transfer_zoo_build_and_forward_smoke(arch_id: str) -> None:
    from dlhub.vision.style_transfer_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
    )

    content = torch.randn(2, 3, 64, 64)
    style = torch.randn(2, 3, 64, 64)
    out = model(content, style)
    assert isinstance(out, dict)
    assert "stylized" in out
    assert tuple(out["stylized"].shape) == (2, 3, 64, 64)

    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()


def test_style_transfer_zoo_script_list_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/style_transfer_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Style transfer local zoo" in list_proc.stdout
    assert "total_arches=" in list_proc.stdout

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/style_transfer_zoo.py", "--smoke", "dlst:style_swap_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: dlst:style_swap_tiny" in smoke_proc.stdout
