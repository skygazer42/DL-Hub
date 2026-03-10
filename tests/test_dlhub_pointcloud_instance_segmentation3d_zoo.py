import pytest

torch = pytest.importorskip("torch")


def test_local_instance_segmentation3d_zoo_lists_90_plus_arches() -> None:
    from dlhub.pointcloud.instance_segmentation3d_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 90

    # A few representative families.
    assert "pcinst3d:bonet_tiny" in arches
    assert "pcinst3d:pointgroup_tiny" in arches
    assert "pcinst3d:softgroup_tiny" in arches
    assert "pcinst3d:mask3d_tiny" in arches
    assert "pcinst3d:mask2former3d_tiny" in arches
    assert "pcinst3d:yolact3d_tiny" in arches


def _tiny_arches() -> list[str]:
    from dlhub.pointcloud.instance_segmentation3d_zoo import list_local_arches

    return [a for a in list_local_arches() if a.split(":", 1)[1].endswith("_tiny")]


@pytest.mark.parametrize("arch_id", _tiny_arches())
def test_local_instance_segmentation3d_zoo_build_and_backward_smoke(arch_id: str) -> None:
    from dlhub.pointcloud.instance_segmentation3d_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=4, width_mult=0.5, dropout=0.0)
    model.train()

    x = torch.randn(2, 128, 3)
    out = model(x)

    assert isinstance(out, dict)
    assert "mask_logits" in out and "cls_logits" in out

    mask_logits = out["mask_logits"]
    cls_logits = out["cls_logits"]
    assert isinstance(mask_logits, torch.Tensor)
    assert isinstance(cls_logits, torch.Tensor)

    assert mask_logits.ndim == 3 and mask_logits.shape[0] == 2 and mask_logits.shape[-1] == 128
    assert cls_logits.ndim == 3 and cls_logits.shape[0] == 2 and cls_logits.shape[-1] == 4
    assert mask_logits.shape[1] == cls_logits.shape[1]

    loss = mask_logits.mean() + cls_logits.mean()
    loss.backward()
