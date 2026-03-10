import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in segmentation smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        (
            "build_unet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "unet_tiny", "dropout": 0.0},
        ),
        (
            "build_deeplabv3plus_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "deeplabv3p_tiny", "width_mult": 0.5},
        ),
        (
            "build_pspnet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "pspnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_fcn_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "fcn8s_tiny", "width_mult": 0.5},
        ),
        (
            "build_deeplabv3_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "deeplabv3_tiny", "width_mult": 0.5},
        ),
        ("build_segnet_segmenter", {"in_channels": 3, "num_classes": 2, "variant": "segnet_tiny"}),
        (
            "build_linknet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "linknet_tiny"},
        ),
        (
            "build_refinenet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "refinenet_tiny", "width_mult": 0.5},
        ),
        (
            "build_enet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "enet_tiny", "width_mult": 0.5},
        ),
        (
            "build_erfnet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "erfnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_espnet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "espnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_espnetv2_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "espnetv2_tiny", "width_mult": 0.5},
        ),
        (
            "build_fastscnn_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "fastscnn_tiny", "width_mult": 0.5},
        ),
        (
            "build_bisenetv1_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "bisenetv1_tiny", "width_mult": 0.5},
        ),
        (
            "build_bisenetv2_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "bisenetv2_tiny", "width_mult": 0.5},
        ),
        (
            "build_icnet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "icnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_cgnet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "cgnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_lednet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "lednet_tiny", "width_mult": 0.5},
        ),
        (
            "build_danet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "danet_tiny", "width_mult": 0.5},
        ),
        (
            "build_ocrnet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "ocrnet_tiny", "width_mult": 0.5},
        ),
        (
            "build_upernet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "upernet_tiny", "width_mult": 0.5},
        ),
        (
            "build_segformer_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "segformer_tiny", "width_mult": 0.5},
        ),
        (
            "build_transunet_segmenter",
            {"in_channels": 3, "num_classes": 2, "variant": "transunet_tiny", "width_mult": 0.5},
        ),
    ],
)
def test_segmentation_algorithms_forward_backward_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.vision.segmentation as seg

    build = getattr(seg, builder_name)
    model = build(**kwargs)
    x = torch.randn(2, int(kwargs["in_channels"]), 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()
