import pytest


torch = pytest.importorskip("torch")


def test_dlhub_vision_detection_models_forward_backward_smoke() -> None:
    from dlhub.vision.detection import (
        build_centernet_detector,
        build_fcos_detector,
        build_retinanet_detector,
        build_yolo_v1_detector,
    )

    x = torch.randn(2, 3, 64, 64)

    fcos = build_fcos_detector(in_channels=3, num_classes=2, variant="fcos_nocenter", width_mult=0.5)
    out = fcos(x)
    assert set(out.keys()) == {"cls_logits", "reg"}
    assert tuple(out["cls_logits"].shape) == (2, 2, 16, 16)
    assert tuple(out["reg"].shape) == (2, 4, 16, 16)
    loss = out["cls_logits"].mean() + out["reg"].mean()
    loss.backward()

    center = build_centernet_detector(in_channels=3, num_classes=2, variant="centernet_tiny", width_mult=0.5)
    out = center(x)
    assert set(out.keys()) == {"heatmap", "wh", "offset"}
    assert tuple(out["heatmap"].shape) == (2, 2, 16, 16)
    assert tuple(out["wh"].shape) == (2, 2, 16, 16)
    assert tuple(out["offset"].shape) == (2, 2, 16, 16)
    loss = out["heatmap"].mean() + out["wh"].mean() + out["offset"].mean()
    loss.backward()

    retina = build_retinanet_detector(in_channels=3, num_classes=3, variant="retinanet_tiny", width_mult=0.5)
    out = retina(x)
    assert set(out.keys()) == {"cls_logits", "bbox_deltas"}
    assert len(out["cls_logits"]) == 3
    assert len(out["bbox_deltas"]) == 3
    for cls, box in zip(out["cls_logits"], out["bbox_deltas"], strict=True):
        assert cls.ndim == 4 and box.ndim == 4
        assert cls.shape[0] == 2 and box.shape[0] == 2
        assert box.shape[1] % 4 == 0
    loss = sum(t.mean() for t in out["cls_logits"]) + sum(t.mean() for t in out["bbox_deltas"])
    loss.backward()

    yolo = build_yolo_v1_detector(in_channels=3, num_classes=2, variant="yolo_v1_tiny", width_mult=0.5)
    out = yolo(x)
    assert set(out.keys()) == {"obj_logits", "cls_logits", "bbox"}
    assert tuple(out["obj_logits"].shape) == (2, 1, 16, 16)
    assert tuple(out["cls_logits"].shape) == (2, 2, 16, 16)
    assert tuple(out["bbox"].shape) == (2, 4, 16, 16)
    loss = out["obj_logits"].mean() + out["cls_logits"].mean() + out["bbox"].mean()
    loss.backward()


def test_dlhub_vision_segmentation_models_forward_backward_smoke() -> None:
    from dlhub.vision.segmentation import build_deeplabv3plus_segmenter, build_pspnet_segmenter, build_unet_segmenter

    x = torch.randn(2, 3, 64, 64)
    for build in [
        lambda: build_unet_segmenter(in_channels=3, num_classes=2, variant="unet_tiny", dropout=0.0),
        lambda: build_deeplabv3plus_segmenter(in_channels=3, num_classes=2, variant="deeplabv3p_tiny", width_mult=0.5),
        lambda: build_pspnet_segmenter(in_channels=3, num_classes=2, variant="pspnet_tiny", width_mult=0.5),
    ]:
        m = build()
        y = m(x)
        assert tuple(y.shape) == (2, 2, 64, 64)
        loss = y.mean()
        loss.backward()
