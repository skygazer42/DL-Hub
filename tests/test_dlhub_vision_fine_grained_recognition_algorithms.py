import pytest


torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, (list, tuple)):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in fgvc smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        ("build_bilinear_cnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "bilinear_cnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_compact_bilinear_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "compact_bilinear_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_kernel_pooling_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "kernel_pooling_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_lowrank_bilinear_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "lowrank_bilinear_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_hierarchical_bilinear_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "hierarchical_bilinear_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_isqrt_cov_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "isqrt_cov_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_mpn_cov_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "mpn_cov_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_ws_ban_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "ws_ban_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_part_rcnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "part_rcnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_partnet_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "partnet_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_part_stacked_cnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "part_stacked_cnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_pa_cnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "pa_cnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_racnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "racnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_ma_cnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "ma_cnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_dfl_cnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "dfl_cnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_nts_net_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "nts_net_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_tasn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "tasn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_s3n_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "s3n_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_mge_cnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "mge_cnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_pmg_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "pmg_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_osme_mamc_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "osme_mamc_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_api_net_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "api_net_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_crossx_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "crossx_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_region_grouping_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "region_grouping_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_dcl_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "dcl_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_ws_dan_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "ws_dan_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_proto_pnet_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "proto_pnet_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_hse_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "hse_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_interp_parts_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "interp_parts_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_ga_cnn_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "ga_cnn_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_transfg_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "transfg_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_ffvt_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "ffvt_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_pedtrans_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "pedtrans_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_vit_fod_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "vit_fod_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_aftrans_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "aftrans_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_sim_trans_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "sim_trans_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_pca_net_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "pca_net_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_metaformer_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "metaformer_fgvc_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_pim_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "pim_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
        ("build_cvl_fgvc_classifier", {"in_channels": 3, "num_classes": 5, "variant": "cvl_tiny", "image_size": 64, "width_mult": 0.5, "dropout": 0.0}),
    ],
)
def test_fgvc_algorithms_forward_backward_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.vision.fine_grained_recognition as fgvc

    build = getattr(fgvc, builder_name)
    model = build(**kwargs)
    x = torch.randn(2, int(kwargs["in_channels"]), int(kwargs["image_size"]), int(kwargs["image_size"]))
    out = model(x)
    assert isinstance(out, dict)
    assert "logits" in out
    assert tuple(out["logits"].shape) == (2, int(kwargs["num_classes"]))
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()
