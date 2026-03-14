import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in vision MOT smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        (
            "build_sort_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "sort_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_deepsort_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "deepsort_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_bytetrack_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "bytetrack_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_ocsort_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "ocsort_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_jde_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "jde_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_fairmot_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "fairmot_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_transtrack_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "transtrack_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_network_flow_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "network_flow_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_motdt_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "motdt_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_motrv2_tracker",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "image_size": 64,
                "variant": "motrv2_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
    ],
)
def test_vision_mot_algorithms_track_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.vision.mot as mot

    build = getattr(mot, builder_name)
    tracker = build(**kwargs)
    video = torch.randn(
        2,
        int(kwargs["seq_len"]),
        int(kwargs["in_channels"]),
        int(kwargs["image_size"]),
        int(kwargs["image_size"]),
    )
    out = tracker.track(video)
    assert isinstance(out, dict)
    assert "track_boxes" in out
    assert "track_scores" in out
    assert "track_ids" in out
    boxes = out["track_boxes"]
    scores = out["track_scores"]
    track_ids = out["track_ids"]
    assert boxes.ndim == 3 and boxes.shape[0] == 2 and boxes.shape[-1] == 4
    assert scores.ndim == 2 and scores.shape[0] == 2
    assert track_ids.ndim == 2 and track_ids.shape[0] == 2
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
