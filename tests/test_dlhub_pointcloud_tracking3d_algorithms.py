import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in tracking3d smoke: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        (
            "build_ab3dmot_tracker3d",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "variant": "ab3dmot_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_centerpoint_track_tracker3d",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "variant": "centerpoint_track_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_simpletrack_tracker3d",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "variant": "simpletrack_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_bitrack_tracker3d",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "variant": "bitrack_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_motsf3d_tracker3d",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "variant": "motsf3d_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_imm_kalman_tracker3d",
            {
                "in_channels": 3,
                "num_classes": 3,
                "seq_len": 4,
                "variant": "imm_kalman_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
    ],
)
def test_tracking3d_algorithms_track_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.pointcloud.tracking3d as tracking3d

    build = getattr(tracking3d, builder_name)
    tracker = build(**kwargs)
    x = torch.randn(2, int(kwargs["seq_len"]), 128, int(kwargs["in_channels"]))
    out = tracker.track(x)
    assert isinstance(out, dict)
    assert "track_boxes" in out
    assert "track_scores" in out
    assert "track_ids" in out
    boxes = out["track_boxes"]
    scores = out["track_scores"]
    track_ids = out["track_ids"]
    assert boxes.ndim == 3 and boxes.shape[0] == 2 and boxes.shape[-1] == 7
    assert scores.ndim == 2 and scores.shape[0] == 2
    assert track_ids.ndim == 2 and track_ids.shape[0] == 2
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
