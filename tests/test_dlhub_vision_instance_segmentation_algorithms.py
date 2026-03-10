import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in instance segmentation smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        (
            "build_yolact_instance_segmenter",
            {
                "in_channels": 3,
                "num_classes": 2,
                "variant": "yolact_tiny",
                "width_mult": 0.5,
                "num_anchors": 3,
            },
        ),
        (
            "build_mask_rcnn_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "mask_rcnn_tiny", "width_mult": 0.5},
        ),
        (
            "build_cascade_mask_rcnn_instance_segmenter",
            {
                "in_channels": 3,
                "num_classes": 2,
                "variant": "cascade_mask_rcnn_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_mask_scoring_rcnn_instance_segmenter",
            {
                "in_channels": 3,
                "num_classes": 2,
                "variant": "mask_scoring_rcnn_tiny",
                "width_mult": 0.5,
            },
        ),
        (
            "build_scnet_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "scnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_pointrend_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "pointrend_tiny", "width_mult": 0.5},
        ),
        (
            "build_htc_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "htc_tiny", "width_mult": 0.5},
        ),
        (
            "build_fcis_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "fcis_tiny", "width_mult": 0.5},
        ),
        (
            "build_solo_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "solo_tiny", "width_mult": 0.5},
        ),
        (
            "build_solov2_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "solov2_tiny", "width_mult": 0.5},
        ),
        (
            "build_condinst_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "condinst_tiny", "width_mult": 0.5},
        ),
        (
            "build_boxinst_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "boxinst_tiny", "width_mult": 0.5},
        ),
        (
            "build_centermask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "centermask_tiny", "width_mult": 0.5},
        ),
        (
            "build_polarmask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "polarmask_tiny", "width_mult": 0.5},
        ),
        (
            "build_tensormask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "tensormask_tiny", "width_mult": 0.5},
        ),
        (
            "build_sparseinst_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "sparseinst_tiny", "width_mult": 0.5},
        ),
        (
            "build_queryinst_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "queryinst_tiny", "width_mult": 0.5},
        ),
        (
            "build_maskformer_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "maskformer_tiny", "width_mult": 0.5},
        ),
        (
            "build_mask2former_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "mask2former_tiny", "width_mult": 0.5},
        ),
        (
            "build_detr_mask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "detr_mask_tiny", "width_mult": 0.5},
        ),
        (
            "build_blendmask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "blendmask_tiny", "width_mult": 0.5},
        ),
        (
            "build_bcnet_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "bcnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_cfm_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "cfm_tiny", "width_mult": 0.5},
        ),
        (
            "build_dct_mask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "dct_mask_tiny", "width_mult": 0.5},
        ),
        (
            "build_deepmask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "deepmask_tiny", "width_mult": 0.5},
        ),
        (
            "build_sipmask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "sipmask_tiny", "width_mult": 0.5},
        ),
        (
            "build_mask_dino_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "mask_dino_tiny", "width_mult": 0.5},
        ),
        (
            "build_deepsnake_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "deepsnake_tiny", "width_mult": 0.5},
        ),
        (
            "build_dynamicinst_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "dynamicinst_tiny", "width_mult": 0.5},
        ),
        (
            "build_e2ec_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "e2ec_tiny", "width_mult": 0.5},
        ),
        (
            "build_fastinst_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "fastinst_tiny", "width_mult": 0.5},
        ),
        (
            "build_instancefcn_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "instancefcn_tiny", "width_mult": 0.5},
        ),
        (
            "build_meinst_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "meinst_tiny", "width_mult": 0.5},
        ),
        (
            "build_mnc_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "mnc_tiny", "width_mult": 0.5},
        ),
        (
            "build_orienmask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "orienmask_tiny", "width_mult": 0.5},
        ),
        (
            "build_panet_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "panet_tiny", "width_mult": 0.5},
        ),
        (
            "build_refinemask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "refinemask_tiny", "width_mult": 0.5},
        ),
        (
            "build_rtmdet_ins_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "rtmdet_ins_tiny", "width_mult": 0.5},
        ),
        (
            "build_shapemask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "shapemask_tiny", "width_mult": 0.5},
        ),
        (
            "build_sharpmask_instance_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "sharpmask_tiny", "width_mult": 0.5},
        ),
    ],
)
def test_instance_segmentation_algorithms_forward_backward_smoke(
    builder_name: str, kwargs: dict
) -> None:
    import dlhub.vision.instance_segmentation as inst

    build = getattr(inst, builder_name)
    model = build(**kwargs)
    x = torch.randn(2, int(kwargs["in_channels"]), 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()
