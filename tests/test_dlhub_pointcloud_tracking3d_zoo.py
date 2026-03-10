import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in tracking3d zoo smoke: {type(x)!r}")


def test_tracking3d_zoo_lists_first_batch_arches() -> None:
    from dlhub.pointcloud.tracking3d_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 18
    assert "pctrk3d:ab3dmot_tiny" in arches
    assert "pctrk3d:centerpoint_track_tiny" in arches
    assert "pctrk3d:simpletrack_tiny" in arches
    assert "pctrk3d:bitrack_tiny" in arches
    assert "pctrk3d:motsf3d_tiny" in arches
    assert "pctrk3d:imm_kalman_tiny" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "pctrk3d:ab3dmot_tiny",
        "pctrk3d:centerpoint_track_tiny",
        "pctrk3d:simpletrack_tiny",
        "pctrk3d:bitrack_tiny",
    ],
)
def test_tracking3d_zoo_build_and_track_smoke(arch_id: str) -> None:
    from dlhub.pointcloud.tracking3d_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        num_classes=3,
        seq_len=4,
        width_mult=0.5,
        dropout=0.0,
    )
    x = torch.randn(2, 4, 128, 3)
    out = model.track(x)
    assert isinstance(out, dict)
    assert "track_boxes" in out and "track_scores" in out and "track_ids" in out
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
