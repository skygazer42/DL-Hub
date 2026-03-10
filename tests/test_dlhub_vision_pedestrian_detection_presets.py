import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type: {type(x)!r}")


@pytest.mark.parametrize(
    "arch_id",
    [
        "dldet:pedestrian_fcos",
        "dldet:pedestrian_retinanet",
        "dldet:pedestrian_faster_rcnn",
        "dldet:pedestrian_ssd",
        "dldet:pedestrian_yolov5",
        "dldet:pedestrian_yolov8",
        "dldet:pedestrian_yolox",
        "dldet:pedestrian_rtdetr",
    ],
)
def test_pedestrian_presets_forward_backward_smoke(arch_id: str) -> None:
    from dlhub.vision.detection_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=1, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()

