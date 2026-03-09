import pytest


torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, (list, tuple)):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in detection smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        # Existing detection families.
        ("build_fcos_detector", {"in_channels": 3, "num_classes": 2, "variant": "fcos_nocenter", "width_mult": 0.5}),
        ("build_centernet_detector", {"in_channels": 3, "num_classes": 2, "variant": "centernet_tiny", "width_mult": 0.5}),
        ("build_retinanet_detector", {"in_channels": 3, "num_classes": 2, "variant": "retinanet_tiny", "width_mult": 0.5, "num_anchors": 3}),
        ("build_yolo_v1_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolo_v1_tiny", "width_mult": 0.5}),
        # Planned additions (should exist after the 40-algorithms rollout).
        ("build_ssd_detector", {"in_channels": 3, "num_classes": 2, "variant": "ssd_tiny", "width_mult": 0.5}),
        ("build_dssd_detector", {"in_channels": 3, "num_classes": 2, "variant": "dssd_tiny", "width_mult": 0.5}),
        ("build_efficientdet_detector", {"in_channels": 3, "num_classes": 2, "variant": "efficientdet_tiny", "width_mult": 0.5}),
        ("build_squeezedet_detector", {"in_channels": 3, "num_classes": 2, "variant": "squeezedet_tiny", "width_mult": 0.5}),
        ("build_yolov2_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolov2_tiny", "width_mult": 0.5}),
        ("build_yolov3_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolov3_tiny", "width_mult": 0.5}),
        ("build_yolov5_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolov5_tiny", "width_mult": 0.5}),
        ("build_yolov8_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolov8_tiny", "width_mult": 0.5}),
        ("build_yolox_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolox_tiny", "width_mult": 0.5}),
        ("build_yolof_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolof_tiny", "width_mult": 0.5}),
        ("build_ppyoloe_detector", {"in_channels": 3, "num_classes": 2, "variant": "ppyoloe_tiny", "width_mult": 0.5}),
        ("build_rtmdet_detector", {"in_channels": 3, "num_classes": 2, "variant": "rtmdet_tiny", "width_mult": 0.5}),
        ("build_nanodet_detector", {"in_channels": 3, "num_classes": 2, "variant": "nanodet_tiny", "width_mult": 0.5}),
        ("build_gfl_detector", {"in_channels": 3, "num_classes": 2, "variant": "gfl_tiny", "width_mult": 0.5}),
        ("build_tood_detector", {"in_channels": 3, "num_classes": 2, "variant": "tood_tiny", "width_mult": 0.5}),
        ("build_varifocalnet_detector", {"in_channels": 3, "num_classes": 2, "variant": "varifocalnet_tiny", "width_mult": 0.5}),
        ("build_vfnet_detector", {"in_channels": 3, "num_classes": 2, "variant": "vfnet_tiny", "width_mult": 0.5}),
        ("build_atss_detector", {"in_channels": 3, "num_classes": 2, "variant": "atss_tiny", "width_mult": 0.5}),
        ("build_paa_detector", {"in_channels": 3, "num_classes": 2, "variant": "paa_tiny", "width_mult": 0.5}),
        ("build_freeanchor_detector", {"in_channels": 3, "num_classes": 2, "variant": "freeanchor_tiny", "width_mult": 0.5}),
        ("build_fsaf_detector", {"in_channels": 3, "num_classes": 2, "variant": "fsaf_tiny", "width_mult": 0.5}),
        ("build_reppoints_detector", {"in_channels": 3, "num_classes": 2, "variant": "reppoints_tiny", "width_mult": 0.5}),
        ("build_foveabox_detector", {"in_channels": 3, "num_classes": 2, "variant": "foveabox_tiny", "width_mult": 0.5}),
        ("build_cornernet_detector", {"in_channels": 3, "num_classes": 2, "variant": "cornernet_tiny", "width_mult": 0.5}),
        ("build_extremenet_detector", {"in_channels": 3, "num_classes": 2, "variant": "extremenet_tiny", "width_mult": 0.5}),
        ("build_ttfnet_detector", {"in_channels": 3, "num_classes": 2, "variant": "ttfnet_tiny", "width_mult": 0.5}),
        ("build_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "detr_tiny", "width_mult": 0.5}),
        ("build_deformable_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "deformable_detr_tiny", "width_mult": 0.5}),
        ("build_conditional_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "conditional_detr_tiny", "width_mult": 0.5}),
        ("build_dab_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "dab_detr_tiny", "width_mult": 0.5}),
        ("build_dn_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "dn_detr_tiny", "width_mult": 0.5}),
        ("build_dino_detector", {"in_channels": 3, "num_classes": 2, "variant": "dino_tiny", "width_mult": 0.5}),
        ("build_rtdetr_detector", {"in_channels": 3, "num_classes": 2, "variant": "rtdetr_tiny", "width_mult": 0.5}),
        ("build_sparse_rcnn_detector", {"in_channels": 3, "num_classes": 2, "variant": "sparse_rcnn_tiny", "width_mult": 0.5}),
        ("build_faster_rcnn_detector", {"in_channels": 3, "num_classes": 2, "variant": "faster_rcnn_tiny", "width_mult": 0.5}),
        ("build_mask_rcnn_detector", {"in_channels": 3, "num_classes": 2, "variant": "mask_rcnn_tiny", "width_mult": 0.5}),
        ("build_cascade_rcnn_detector", {"in_channels": 3, "num_classes": 2, "variant": "cascade_rcnn_tiny", "width_mult": 0.5}),
        ("build_rfcn_detector", {"in_channels": 3, "num_classes": 2, "variant": "rfcn_tiny", "width_mult": 0.5}),
        # 50-family archive expansion representatives.
        ("build_overfeat_detector", {"in_channels": 3, "num_classes": 2, "variant": "overfeat_tiny", "width_mult": 0.5}),
        ("build_yolov7_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolov7_tiny", "width_mult": 0.5}),
        ("build_rcnn_detector", {"in_channels": 3, "num_classes": 2, "variant": "rcnn_tiny", "width_mult": 0.5}),
        ("build_grid_rcnn_detector", {"in_channels": 3, "num_classes": 2, "variant": "grid_rcnn_tiny", "width_mult": 0.5}),
        ("build_densebox_detector", {"in_channels": 3, "num_classes": 2, "variant": "densebox_tiny", "width_mult": 0.5}),
        ("build_centernet2_detector", {"in_channels": 3, "num_classes": 2, "variant": "centernet2_tiny", "width_mult": 0.5}),
        ("build_anchor_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "anchor_detr_tiny", "width_mult": 0.5}),
        ("build_co_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "co_detr_tiny", "width_mult": 0.5}),
        ("build_glip_detector", {"in_channels": 3, "num_classes": 2, "variant": "glip_tiny", "width_mult": 0.5}),
        ("build_yolo_world_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolo_world_tiny", "width_mult": 0.5}),
        # Recent archive expansion representatives.
        ("build_yolo11_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolo11_tiny", "width_mult": 0.5}),
        ("build_yolo13_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolo13_tiny", "width_mult": 0.5}),
        ("build_d_fine_detector", {"in_channels": 3, "num_classes": 2, "variant": "d_fine_tiny", "width_mult": 0.5}),
        ("build_lw_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "lw_detr_tiny", "width_mult": 0.5}),
        ("build_ovlw_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "ovlw_detr_tiny", "width_mult": 0.5}),
        ("build_rtgen_detector", {"in_channels": 3, "num_classes": 2, "variant": "rtgen_tiny", "width_mult": 0.5}),
        ("build_sa_detr_detector", {"in_channels": 3, "num_classes": 2, "variant": "sa_detr_tiny", "width_mult": 0.5}),
        ("build_yolo26_detector", {"in_channels": 3, "num_classes": 2, "variant": "yolo26_tiny", "width_mult": 0.5}),
    ],
)
def test_detection_algorithms_forward_backward_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.vision.detection as det

    build = getattr(det, builder_name)
    model = build(**kwargs)
    x = torch.randn(2, int(kwargs["in_channels"]), 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()
