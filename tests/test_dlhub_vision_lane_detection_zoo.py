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
    raise TypeError(f"Unsupported output type in lane detection zoo smoke: {type(x)!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_lane_detection_zoo_lists_72_plus_arches() -> None:
    from dlhub.vision.lane_detection_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 72
    assert "dllane:lanenet_tiny" in arches
    assert "dllane:scnn_tiny" in arches
    assert "dllane:enet_sad_tiny" in arches
    assert "dllane:ufld_tiny" in arches
    assert "dllane:laneatt_tiny" in arches
    assert "dllane:lstr_tiny" in arches
    assert "dllane:resa_tiny" in arches
    assert "dllane:clrnet_tiny" in arches
    assert "dllane:condlanenet_tiny" in arches
    assert "dllane:polylanenet_tiny" in arches
    assert "dllane:bezierlanenet_tiny" in arches
    assert "dllane:pinet_tiny" in arches
    assert "dllane:laneaf_tiny" in arches
    assert "dllane:ganet_tiny" in arches
    assert "dllane:persformer_tiny" in arches
    assert "dllane:lanegcn_tiny" in arches
    assert "dllane:topolane_tiny" in arches
    assert "dllane:bevlanedet_tiny" in arches
    assert "dllane:o2sformer_tiny" in arches
    assert "dllane:latr_tiny" in arches
    assert "dllane:laneformer_tiny" in arches
    assert "dllane:anchor3dlane_tiny" in arches
    assert "dllane:genlanenet_tiny" in arches
    assert "dllane:priorlane_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.vision.lane_detection_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_lane_detection_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.vision.lane_detection_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_rows=16,
        grid_size=32,
        num_anchors=24,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize(
    "arch_id",
    [
        "dllane:lanenet_small",
        "dllane:ufld_base",
        "dllane:lstr_small",
        "dllane:clrnet_small",
        "dllane:polylanenet_base",
        "dllane:pinet_small",
        "dllane:persformer_base",
        "dllane:topolane_small",
        "dllane:latr_base",
        "dllane:laneformer_small",
        "dllane:anchor3dlane_base",
    ],
)
def test_lane_detection_zoo_builds_non_tiny_variants(arch_id: str) -> None:
    from dlhub.vision.lane_detection_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_rows=16,
        grid_size=32,
        num_anchors=24,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)


def test_lane_detection_zoo_script_list_and_smoke() -> None:
    list_proc = subprocess.run(
        [sys.executable, "scripts/lane_detection_zoo.py", "--list", "--limit", "8"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert list_proc.returncode == 0
    assert "Lane detection local zoo" in list_proc.stdout
    assert "total_arches=" in list_proc.stdout

    smoke_proc = subprocess.run(
        [
            sys.executable,
            "scripts/lane_detection_zoo.py",
            "--smoke",
            "dllane:lstr_tiny",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_proc.returncode == 0
    assert "smoke: dllane:lstr_tiny" in smoke_proc.stdout


def test_lane_detection_zoo_clrnet_accepts_non_half_width_multiplier() -> None:
    from dlhub.vision.lane_detection_zoo import build_local_model

    model = build_local_model(
        "dllane:clrnet_small",
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=1.1,
        dropout=0.0,
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert tuple(out["lane_logits"].shape) == (1, 6)


def test_bezierlanenet_default_num_points_matches_zoo_default() -> None:
    from dlhub.vision.lane_detection import build_bezierlanenet_lane_detector
    from dlhub.vision.lane_detection_zoo import build_local_model

    direct = build_bezierlanenet_lane_detector(in_channels=3, num_lanes=4)
    via_zoo = build_local_model("dllane:bezierlanenet_small", in_channels=3, num_lanes=4)

    x = torch.randn(1, 3, 64, 64)
    direct_out = direct(x)
    zoo_out = via_zoo(x)
    assert direct_out["control_points"].shape[2] == zoo_out["control_points"].shape[2] == 16
