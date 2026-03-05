import pytest


torch = pytest.importorskip("torch")


def test_local_segmentation3d_zoo_lists_120_plus_arches() -> None:
    from dlhub.pointcloud.segmentation3d_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 120

    # A few representative families.
    assert "pcseg3d:pointnet_tiny" in arches
    assert "pcseg3d:pointnet2_tiny" in arches
    assert "pcseg3d:dgcnn_tiny" in arches
    assert "pcseg3d:kpconv_tiny" in arches
    assert "pcseg3d:randlanet_tiny" in arches
    assert "pcseg3d:cylinder3d_tiny" in arches
    assert "pcseg3d:minkunet_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.pointcloud.segmentation3d_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_local_segmentation3d_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.pointcloud.segmentation3d_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=6, width_mult=0.5, dropout=0.0)
    model.train()

    x = torch.randn(2, 128, 3)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (2, 128, 6)

    loss = y.mean()
    loss.backward()

