import pytest


torch = pytest.importorskip("torch")


def test_local_detection3d_zoo_lists_120_plus_arches() -> None:
    from dlhub.pointcloud.detection3d_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 120

    # A few representative families.
    assert "pcdet3d:pointpillars_tiny" in arches
    assert "pcdet3d:pv_rcnn_tiny" in arches
    assert "pcdet3d:point_rcnn_tiny" in arches
    assert "pcdet3d:votenet_tiny" in arches
    assert "pcdet3d:threedetr_tiny" in arches
    assert "pcdet3d:fcaf3d_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.pointcloud.detection3d_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_local_detection3d_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.pointcloud.detection3d_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=3, width_mult=0.5, dropout=0.0)
    model.train()

    x = torch.randn(2, 128, 3)
    out = model(x)

    assert isinstance(out, dict)
    assert "boxes" in out and "cls_logits" in out

    boxes = out["boxes"]
    logits = out["cls_logits"]
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(logits, torch.Tensor)
    assert boxes.ndim == 3 and boxes.shape[0] == 2 and boxes.shape[-1] == 7
    assert logits.ndim == 3 and logits.shape[0] == 2 and logits.shape[-1] == 3

    loss = boxes.mean() + logits.mean()
    loss.backward()

