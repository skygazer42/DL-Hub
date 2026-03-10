import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in panoptic segmentation smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        (
            "build_panoptic_fpn_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "panoptic_fpn_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_upsnet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "upsnet_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_aunet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "aunet_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_tascnet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "tascnet_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_efficientps_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "efficientps_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_panoptic_deeplab_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "panoptic_deeplab_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_panoptic_fcn_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "panoptic_fcn_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_axial_deeplab_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "axial_deeplab_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_yolact_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "yolact_panoptic_tiny",
                "width_mult": 0.5,
                "num_anchors": 3,
            },
        ),
        (
            "build_blendmask_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "blendmask_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_condinst_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "condinst_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_boxinst_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "boxinst_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_centermask_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "centermask_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_polarmask_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "polarmask_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_tensormask_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "tensormask_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_sparseinst_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "sparseinst_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_queryinst_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "queryinst_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_solo_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "solo_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_solov2_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "solov2_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_scnet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "scnet_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_pointrend_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "pointrend_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_maskformer_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "maskformer_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_mask2former_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "mask2former_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_knet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "knet_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_detr_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "detr_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_dn_detr_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "dn_detr_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_deformable_detr_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "deformable_detr_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_conditional_detr_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "conditional_detr_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_dab_detr_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "dab_detr_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_sparse_rcnn_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "sparse_rcnn_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_rtdetr_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "rtdetr_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_panoptic_segformer_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "panoptic_segformer_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_upernet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "upernet_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_transunet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "transunet_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_uberpanoptic_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "uberpanoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_max_deeplab_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "max_deeplab_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_setr_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "setr_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_hrnet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "hrnet_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_bisenet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "bisenet_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_ocrnet_panoptic_segmenter",
            {
                "in_channels": 3,
                "num_thing_classes": 3,
                "num_stuff_classes": 2,
                "variant": "ocrnet_panoptic_tiny",
                "width_mult": 0.5,
            },
        ),
    ],
)
def test_panoptic_segmentation_algorithms_forward_backward_smoke(
    builder_name: str, kwargs: dict
) -> None:
    import dlhub.vision.panoptic_segmentation as pan

    build = getattr(pan, builder_name)
    model = build(**kwargs)
    x = torch.randn(2, int(kwargs["in_channels"]), 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()
