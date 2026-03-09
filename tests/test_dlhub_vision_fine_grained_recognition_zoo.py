
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
    assert len(arches) >= 216
    assert "dlfgvc:bilinear_cnn_tiny" in arches
    assert "dlfgvc:racnn_tiny" in arches
    assert "dlfgvc:ws_dan_tiny" in arches
    assert "dlfgvc:transfg_tiny" in arches
    assert "dlfgvc:metaformer_fgvc_tiny" in arches
    assert "dlfgvc:vpt_tiny" in arches
    assert "dlfgvc:sm_vit_tiny" in arches
    assert "dlfgvc:ldh_vit_tiny" in arches
    assert "dlfgvc:prompt_cam_tiny" in arches
    assert "dlfgvc:fg_clip_tiny" in arches
    assert "dlfgvc:finer_cam_tiny" in arches
    assert "dlfgvc:xr_vlm_tiny" in arches
    assert "dlfgvc:img_cot_tiny" in arches
    assert "dlfgvc:refine_rft_tiny" in arches
    assert "dlfgvc:iir_vlm_tiny" in arches
    assert "dlfgvc:fine_r1_tiny" in arches
    assert "dlfgvc:r2i_distill_tiny" in arches
    assert "dlfgvc:gft_tiny" in arches
    assert "dlfgvc:e_finer_tiny" in arches
    assert "dlfgvc:unifgvc_tiny" in arches
    assert "dlfgvc:granvit_tiny" in arches
    assert "dlfgvc:saccadic_vision_tiny" in arches
    assert "dlfgvc:causal_fsfg_tiny" in arches
    assert "dlfgvc:micro_clip_tiny" in arches
    assert "dlfgvc:dcnn_fg_tiny" in arches
    assert "dlfgvc:hfcr_net_tiny" in arches
    assert "dlfgvc:ficnet_tiny" in arches
    assert "dlfgvc:cmcp_meta_tiny" in arches
    assert "dlfgvc:gcpl_tiny" in arches
    assert "dlfgvc:comple_tiny" in arches
    assert "dlfgvc:pp_ssl_tiny" in arches
    assert "dlfgvc:part_rel_transformer_tiny" in arches
    assert "dlfgvc:highorder_graph_tiny" in arches
    assert "dlfgvc:part_matching_tiny" in arches
    assert "dlfgvc:saliency_partition_tiny" in arches
    assert "dlfgvc:late_fusion_transformer_tiny" in arches


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


@pytest.mark.parametrize(
    "arch_id",
    [
        "dlfgvc:gft_small",
        "dlfgvc:e_finer_base",
        "dlfgvc:unifgvc_small",
        "dlfgvc:granvit_small",
        "dlfgvc:micro_clip_small",
        "dlfgvc:cmcp_meta_small",
        "dlfgvc:gcpl_small",
        "dlfgvc:comple_base",
        "dlfgvc:highorder_graph_small",
        "dlfgvc:part_matching_small",
    ],
)
def test_fgvc_recent_zoo_builds_non_tiny_variants(arch_id: str) -> None:
    from dlhub.vision.fine_grained_recognition_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=5, image_size=64, width_mult=0.5, dropout=0.0)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert "logits" in out
    assert tuple(out["logits"].shape) == (1, 5)


def test_fgvc_zoo_script_list_and_smoke() -> None:
    import re

    list_proc = subprocess.run(
        [sys.executable, "scripts/fine_grained_recognition_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Fine-grained recognition local zoo" in list_proc.stdout
    m = re.search(r"total_arches=(\d+)", list_proc.stdout)
    assert m is not None
    assert int(m.group(1)) >= 216

    smoke_proc = subprocess.run(
        [sys.executable, "scripts/fine_grained_recognition_zoo.py", "--smoke", "dlfgvc:transfg_tiny"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: dlfgvc:transfg_tiny" in smoke_proc.stdout


def test_fgvc_zoo_script_timeline() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/fine_grained_recognition_zoo.py", "--timeline"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Fine-grained recognition timeline" in proc.stdout
    # Spot-check a few entries so the flag keeps working over time.
    assert "\n2015\n" in proc.stdout
    assert "bilinear_cnn" in proc.stdout
    assert "fine_r1" in proc.stdout
    assert "gft" in proc.stdout
    assert "causal_fsfg" in proc.stdout
    assert "micro_clip" in proc.stdout
    assert "cmcp_meta" in proc.stdout
    assert "gcpl" in proc.stdout
    assert "comple" in proc.stdout
    assert "pp_ssl" in proc.stdout
    assert "part_rel_transformer" in proc.stdout
    assert "highorder_graph" in proc.stdout
    assert "part_matching" in proc.stdout
    assert "saliency_partition" in proc.stdout
    assert "late_fusion_transformer" in proc.stdout
