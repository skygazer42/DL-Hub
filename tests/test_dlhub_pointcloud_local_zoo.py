import pytest

torch = pytest.importorskip("torch")


def test_local_pointcloud_zoo_lists_30_plus_arches() -> None:
    from dlhub.pointcloud.local_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 30

    # A few representative families.
    assert "pc:pointnet" in arches
    assert "pc:pointnet_tnet" in arches
    assert "pc:pointnet2_ssg" in arches
    assert "pc:dgcnn" in arches
    assert "pc:pointgcn" in arches
    assert "pc:pointgat" in arches
    assert "pc:pointweb" in arches
    assert "pc:spidercnn" in arches
    assert "pc:rscnn" in arches
    assert "pc:paconv" in arches
    assert "pc:curvenet" in arches
    assert "pc:gdanet" in arches
    assert "pc:pointsift" in arches
    assert "pc:point2seq" in arches
    assert "pc:asnl" in arches
    assert "pc:randlanet" in arches
    assert "pc:pvcnn" in arches
    assert "pc:point_transformer" in arches
    assert "pc:pct" in arches
    assert "pc:pointbert" in arches
    assert "pc:pointmae" in arches
    assert "pc:pointmixer" in arches
    assert "pc:simpleview" in arches
    assert "pc:pointmlp" in arches
    assert "pc:pointnext_tiny" in arches
    assert "pc:pointcnn" in arches
    assert "pc:kpconv" in arches
    assert "pc:shellnet" in arches


@pytest.mark.parametrize(
    ("arch_id", "width_mult"),
    [
        ("pc:pointnet", 1.0),
        ("pc:dgcnn", 0.5),
        ("pc:pointnet2_ssg", 0.5),
        ("pc:point_transformer", 0.5),
        ("pc:pointmlp", 0.5),
    ],
)
def test_local_pointcloud_zoo_build_smoke(arch_id: str, width_mult: float) -> None:
    from dlhub.pointcloud.local_zoo import build_local_model

    model = build_local_model(
        arch_id, in_channels=3, num_classes=4, num_points=64, width_mult=float(width_mult)
    )
    model.eval()

    x = torch.zeros(2, 64, 3)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (2, 4)
