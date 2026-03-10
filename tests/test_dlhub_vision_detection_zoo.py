import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in detection zoo smoke: {type(x)!r}")


def test_detection_zoo_list_and_build_smoke() -> None:
    from dlhub.vision.detection_zoo import build_local_model, list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 304
    assert "dldet:ssd_tiny" in arches
    assert "dldet:detr_tiny" in arches
    assert "dldet:dino_tiny" in arches
    assert "dldet:yolov8_tiny" in arches
    assert "dldet:yolov4_tiny" in arches
    assert "dldet:yolov10_tiny" in arches
    assert "dldet:rcnn_tiny" in arches
    assert "dldet:densebox_tiny" in arches
    assert "dldet:anchor_detr_tiny" in arches
    assert "dldet:glip_tiny" in arches
    assert "dldet:yolo_world_tiny" in arches
    assert "dldet:yolo11_tiny" in arches
    assert "dldet:yolo13_tiny" in arches
    assert "dldet:d_fine_tiny" in arches
    assert "dldet:lw_detr_tiny" in arches
    assert "dldet:ovlw_detr_tiny" in arches
    assert "dldet:rtgen_tiny" in arches
    assert "dldet:sa_detr_tiny" in arches
    assert "dldet:yolo26_tiny" in arches

    for arch_id in [
        "dldet:ssd_tiny",
        "dldet:detr_tiny",
        "dldet:dino_tiny",
        "dldet:yolov8_tiny",
        "dldet:yolo_v1_tiny",
        "dldet:yolov4_tiny",
        "dldet:rcnn_tiny",
        "dldet:densebox_tiny",
        "dldet:anchor_detr_tiny",
        "dldet:glip_tiny",
        "dldet:yolo11_tiny",
        "dldet:d_fine_tiny",
        "dldet:lw_detr_tiny",
        "dldet:rtgen_tiny",
        "dldet:yolo26_tiny",
    ]:
        model = build_local_model(arch_id, in_channels=3, num_classes=2, width_mult=0.5)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        loss = _sum_tensor_means(out)
        assert torch.isfinite(loss)
        loss.backward()
