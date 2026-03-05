import pytest


torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, (list, tuple)):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in detection zoo smoke: {type(x)!r}")


def test_detection_zoo_list_and_build_smoke() -> None:
    from dlhub.vision.detection_zoo import build_local_model, list_local_arches

    arches = list_local_arches()
    assert "dldet:ssd_tiny" in arches
    assert "dldet:detr_tiny" in arches

    for arch_id in ["dldet:ssd_tiny", "dldet:detr_tiny", "dldet:yolo_v1_tiny"]:
        model = build_local_model(arch_id, in_channels=3, num_classes=2, width_mult=0.5)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        loss = _sum_tensor_means(out)
        assert torch.isfinite(loss)
        loss.backward()

