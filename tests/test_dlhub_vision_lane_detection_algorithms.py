import warnings

import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported lane detection output type: {type(x)!r}")


@pytest.mark.parametrize(
    ("builder_name", "variant"),
    [
        ("build_lanenet_lane_detector", "lanenet_tiny"),
        ("build_scnn_lane_detector", "scnn_tiny"),
        ("build_enet_sad_lane_detector", "enet_sad_tiny"),
        ("build_ufld_lane_detector", "ufld_tiny"),
        ("build_laneatt_lane_detector", "laneatt_tiny"),
        ("build_lstr_lane_detector", "lstr_tiny"),
        ("build_resa_lane_detector", "resa_tiny"),
        ("build_clrnet_lane_detector", "clrnet_tiny"),
        ("build_condlanenet_lane_detector", "condlanenet_tiny"),
        ("build_polylanenet_lane_detector", "polylanenet_tiny"),
        ("build_bezierlanenet_lane_detector", "bezierlanenet_tiny"),
        ("build_pinet_lane_detector", "pinet_tiny"),
        ("build_laneaf_lane_detector", "laneaf_tiny"),
        ("build_ganet_lane_detector", "ganet_tiny"),
        ("build_persformer_lane_detector", "persformer_tiny"),
        ("build_lanegcn_lane_detector", "lanegcn_tiny"),
        ("build_topolane_lane_detector", "topolane_tiny"),
        ("build_bevlanedet_lane_detector", "bevlanedet_tiny"),
        ("build_o2sformer_lane_detector", "o2sformer_tiny"),
        ("build_latr_lane_detector", "latr_tiny"),
        ("build_laneformer_lane_detector", "laneformer_tiny"),
        ("build_anchor3dlane_lane_detector", "anchor3dlane_tiny"),
        ("build_genlanenet_lane_detector", "genlanenet_tiny"),
        ("build_priorlane_lane_detector", "priorlane_tiny"),
    ],
)
def test_lane_detection_algorithms_forward_backward_smoke(
    builder_name: str,
    variant: str,
) -> None:
    import dlhub.vision.lane_detection as lane

    build = getattr(lane, builder_name)
    model = build(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_rows=16,
        grid_size=32,
        num_anchors=24,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant=variant,
    )

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()


def test_lanenet_family_outputs_binary_and_embedding_maps() -> None:
    from dlhub.vision.lane_detection import build_lanenet_lane_detector

    model = build_lanenet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
        variant="lanenet_tiny",
    )

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert set(out) == {"binary_logits", "embedding"}
    assert tuple(out["binary_logits"].shape) == (2, 1, 64, 64)
    assert out["embedding"].shape[0] == 2
    assert out["embedding"].shape[-2:] == (64, 64)


def test_ufld_family_outputs_row_anchor_logits() -> None:
    from dlhub.vision.lane_detection import build_ufld_lane_detector

    model = build_ufld_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_rows=12,
        grid_size=20,
        width_mult=0.5,
        dropout=0.0,
        variant="ufld_tiny",
    )

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert set(out) == {"exist_logits", "row_logits"}
    assert tuple(out["exist_logits"].shape) == (2, 4)
    assert tuple(out["row_logits"].shape) == (2, 4, 12, 20)


def test_laneatt_and_lstr_families_output_structured_lane_predictions() -> None:
    from dlhub.vision.lane_detection import (
        build_laneatt_lane_detector,
        build_lstr_lane_detector,
    )

    laneatt = build_laneatt_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=18,
        num_anchors=28,
        width_mult=0.5,
        dropout=0.0,
        variant="laneatt_tiny",
    )
    lstr = build_lstr_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=18,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="lstr_tiny",
    )

    x = torch.randn(2, 3, 64, 64)
    laneatt_out = laneatt(x)
    assert set(laneatt_out) == {"anchor_embeddings", "cls_logits", "curve_points"}
    assert tuple(laneatt_out["cls_logits"].shape) == (2, 28)
    assert tuple(laneatt_out["curve_points"].shape) == (2, 28, 18, 2)

    lstr_out = lstr(x)
    assert set(lstr_out) == {"curve_points", "lane_logits"}
    assert tuple(lstr_out["lane_logits"].shape) == (2, 6)
    assert tuple(lstr_out["curve_points"].shape) == (2, 6, 18, 2)


def test_resa_family_outputs_binary_logits() -> None:
    from dlhub.vision.lane_detection import build_resa_lane_detector

    model = build_resa_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
        variant="resa_tiny",
    )

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert set(out) == {"binary_logits"}
    assert tuple(out["binary_logits"].shape) == (2, 1, 64, 64)


def test_curve_and_mask_lane_families_output_structured_predictions() -> None:
    from dlhub.vision.lane_detection import (
        build_bezierlanenet_lane_detector,
        build_clrnet_lane_detector,
        build_condlanenet_lane_detector,
        build_polylanenet_lane_detector,
    )

    clrnet = build_clrnet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        width_mult=0.5,
        dropout=0.0,
        variant="clrnet_tiny",
    )
    condlane = build_condlanenet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
        variant="condlanenet_tiny",
    )
    polylane = build_polylanenet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
        variant="polylanenet_tiny",
    )
    bezier = build_bezierlanenet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=5,
        width_mult=0.5,
        dropout=0.0,
        variant="bezierlanenet_tiny",
    )

    x = torch.randn(2, 3, 64, 64)

    clrnet_out = clrnet(x)
    assert set(clrnet_out) == {"curve_points", "lane_logits", "refinement_offsets"}
    assert tuple(clrnet_out["lane_logits"].shape) == (2, 6)
    assert tuple(clrnet_out["curve_points"].shape) == (2, 6, 16, 2)
    assert tuple(clrnet_out["refinement_offsets"].shape) == (2, 6, 16, 2)

    condlane_out = condlane(x)
    assert set(condlane_out) == {"lane_logits", "mask_kernels", "mask_logits"}
    assert tuple(condlane_out["lane_logits"].shape) == (2, 4)
    assert tuple(condlane_out["mask_kernels"].shape[:2]) == (2, 4)
    assert tuple(condlane_out["mask_logits"].shape) == (2, 4, 64, 64)

    polylane_out = polylane(x)
    assert set(polylane_out) == {"lane_logits", "poly_coeffs"}
    assert tuple(polylane_out["lane_logits"].shape) == (2, 4)
    assert tuple(polylane_out["poly_coeffs"].shape) == (2, 4, 4)

    bezier_out = bezier(x)
    assert set(bezier_out) == {"control_points", "lane_logits"}
    assert tuple(bezier_out["lane_logits"].shape) == (2, 4)
    assert tuple(bezier_out["control_points"].shape) == (2, 4, 5, 2)


def test_clrnet_accepts_non_half_width_multiplier() -> None:
    from dlhub.vision.lane_detection import build_clrnet_lane_detector

    model = build_clrnet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=1.1,
        dropout=0.0,
        variant="clrnet_small",
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert tuple(out["lane_logits"].shape) == (1, 6)


def test_pinet_and_laneaf_families_output_embedding_and_affinity_fields() -> None:
    from dlhub.vision.lane_detection import (
        build_laneaf_lane_detector,
        build_pinet_lane_detector,
    )

    pinet = build_pinet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
        variant="pinet_tiny",
    )
    laneaf = build_laneaf_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
        variant="laneaf_tiny",
    )

    x = torch.randn(2, 3, 64, 64)

    pinet_out = pinet(x)
    assert set(pinet_out) == {"confidence_logits", "embedding", "offsets"}
    assert tuple(pinet_out["confidence_logits"].shape) == (2, 1, 64, 64)
    assert pinet_out["embedding"].shape[0] == 2
    assert pinet_out["embedding"].shape[-2:] == (64, 64)
    assert tuple(pinet_out["offsets"].shape) == (2, 2, 64, 64)

    laneaf_out = laneaf(x)
    assert set(laneaf_out) == {"binary_logits", "haf", "vaf"}
    assert tuple(laneaf_out["binary_logits"].shape) == (2, 1, 64, 64)
    assert tuple(laneaf_out["haf"].shape) == (2, 1, 64, 64)
    assert tuple(laneaf_out["vaf"].shape) == (2, 2, 64, 64)


def test_ganet_persformer_and_lanegcn_output_structured_predictions() -> None:
    from dlhub.vision.lane_detection import (
        build_ganet_lane_detector,
        build_lanegcn_lane_detector,
        build_persformer_lane_detector,
    )

    ganet = build_ganet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        width_mult=0.5,
        dropout=0.0,
        variant="ganet_tiny",
    )
    persformer = build_persformer_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="persformer_tiny",
    )
    lanegcn = build_lanegcn_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        width_mult=0.5,
        dropout=0.0,
        variant="lanegcn_tiny",
    )

    x = torch.randn(2, 3, 64, 64)

    ganet_out = ganet(x)
    assert set(ganet_out) == {"binary_logits", "lane_logits", "curve_points"}
    assert tuple(ganet_out["binary_logits"].shape) == (2, 1, 64, 64)
    assert tuple(ganet_out["lane_logits"].shape) == (2, 4)
    assert tuple(ganet_out["curve_points"].shape) == (2, 4, 16, 2)

    persformer_out = persformer(x)
    assert set(persformer_out) == {"lane_logits", "curve_points", "perspective_features"}
    assert tuple(persformer_out["lane_logits"].shape) == (2, 6)
    assert tuple(persformer_out["curve_points"].shape) == (2, 6, 16, 2)
    assert persformer_out["perspective_features"].shape[0] == 2

    lanegcn_out = lanegcn(x)
    assert set(lanegcn_out) == {"adjacency_logits", "curve_points", "lane_logits"}
    assert tuple(lanegcn_out["lane_logits"].shape) == (2, 4)
    assert tuple(lanegcn_out["curve_points"].shape) == (2, 4, 16, 2)
    assert tuple(lanegcn_out["adjacency_logits"].shape) == (2, 4, 4)


def test_topology_and_bev_transformer_lane_families_output_structured_predictions() -> None:
    from dlhub.vision.lane_detection import (
        build_bevlanedet_lane_detector,
        build_latr_lane_detector,
        build_o2sformer_lane_detector,
        build_topolane_lane_detector,
    )

    topolane = build_topolane_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="topolane_tiny",
    )
    bevlane = build_bevlanedet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="bevlanedet_tiny",
    )
    o2sformer = build_o2sformer_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="o2sformer_tiny",
    )
    latr = build_latr_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="latr_tiny",
    )

    x = torch.randn(2, 3, 64, 64)

    topolane_out = topolane(x)
    assert set(topolane_out) == {"curve_points", "lane_logits", "topology_logits"}
    assert tuple(topolane_out["lane_logits"].shape) == (2, 6)
    assert tuple(topolane_out["curve_points"].shape) == (2, 6, 16, 2)
    assert tuple(topolane_out["topology_logits"].shape) == (2, 6, 6)

    bevlane_out = bevlane(x)
    assert set(bevlane_out) == {"bev_features", "curve_points", "lane_logits"}
    assert tuple(bevlane_out["lane_logits"].shape) == (2, 6)
    assert tuple(bevlane_out["curve_points"].shape) == (2, 6, 16, 2)
    assert bevlane_out["bev_features"].shape[0] == 2

    o2sformer_out = o2sformer(x)
    assert set(o2sformer_out) == {"curve_points", "lane_logits", "object_queries"}
    assert tuple(o2sformer_out["lane_logits"].shape) == (2, 6)
    assert tuple(o2sformer_out["curve_points"].shape) == (2, 6, 16, 2)
    assert tuple(o2sformer_out["object_queries"].shape[:2]) == (2, 6)

    latr_out = latr(x)
    assert set(latr_out) == {"anchor_queries", "curve_points", "lane_logits"}
    assert tuple(latr_out["lane_logits"].shape) == (2, 6)
    assert tuple(latr_out["curve_points"].shape) == (2, 6, 16, 2)
    assert tuple(latr_out["anchor_queries"].shape[:2]) == (2, 6)


def test_laneformer_and_3d_structured_lane_families_output_structured_predictions() -> None:
    from dlhub.vision.lane_detection import (
        build_anchor3dlane_lane_detector,
        build_genlanenet_lane_detector,
        build_laneformer_lane_detector,
        build_priorlane_lane_detector,
    )

    laneformer = build_laneformer_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="laneformer_tiny",
    )
    anchor3dlane = build_anchor3dlane_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="anchor3dlane_tiny",
    )
    genlanenet = build_genlanenet_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="genlanenet_tiny",
    )
    priorlane = build_priorlane_lane_detector(
        in_channels=3,
        num_lanes=4,
        image_size=64,
        num_points=16,
        num_queries=6,
        width_mult=0.5,
        dropout=0.0,
        variant="priorlane_tiny",
    )

    x = torch.randn(2, 3, 64, 64)

    laneformer_out = laneformer(x)
    assert set(laneformer_out) == {"curve_points", "lane_logits", "lane_tokens"}
    assert tuple(laneformer_out["lane_logits"].shape) == (2, 6)
    assert tuple(laneformer_out["curve_points"].shape) == (2, 6, 16, 2)
    assert tuple(laneformer_out["lane_tokens"].shape[:2]) == (2, 6)

    anchor3dlane_out = anchor3dlane(x)
    assert set(anchor3dlane_out) == {
        "anchor_embeddings",
        "curve_points",
        "lane_heights",
        "lane_logits",
    }
    assert tuple(anchor3dlane_out["lane_logits"].shape) == (2, 6)
    assert tuple(anchor3dlane_out["curve_points"].shape) == (2, 6, 16, 2)
    assert tuple(anchor3dlane_out["lane_heights"].shape) == (2, 6, 16)
    assert tuple(anchor3dlane_out["anchor_embeddings"].shape[:2]) == (2, 6)

    genlanenet_out = genlanenet(x)
    assert set(genlanenet_out) == {"camera_embedding", "curve_points", "lane_logits"}
    assert tuple(genlanenet_out["lane_logits"].shape) == (2, 6)
    assert tuple(genlanenet_out["curve_points"].shape) == (2, 6, 16, 2)
    assert genlanenet_out["camera_embedding"].shape[0] == 2

    priorlane_out = priorlane(x)
    assert set(priorlane_out) == {"curve_points", "lane_logits", "prior_embeddings"}
    assert tuple(priorlane_out["lane_logits"].shape) == (2, 6)
    assert tuple(priorlane_out["curve_points"].shape) == (2, 6, 16, 2)
    assert tuple(priorlane_out["prior_embeddings"].shape[:2]) == (2, 6)


def test_laneformer_and_genlanenet_do_not_emit_odd_head_transformer_warnings() -> None:
    from dlhub.vision.lane_detection import (
        build_genlanenet_lane_detector,
        build_laneformer_lane_detector,
    )

    x = torch.randn(1, 3, 64, 64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        laneformer = build_laneformer_lane_detector(
            in_channels=3,
            num_lanes=4,
            image_size=64,
            num_points=16,
            num_queries=6,
            width_mult=0.5,
            dropout=0.0,
            variant="laneformer_tiny",
        )
        genlanenet = build_genlanenet_lane_detector(
            in_channels=3,
            num_lanes=4,
            image_size=64,
            num_points=16,
            num_queries=6,
            width_mult=0.5,
            dropout=0.0,
            variant="genlanenet_tiny",
        )
        laneformer(x)
        genlanenet(x)

    odd_head_warnings = [
        str(item.message) for item in caught if "num_heads is odd" in str(item.message)
    ]
    assert odd_head_warnings == []
