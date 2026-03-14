import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in vision MOT zoo smoke: {type(x)!r}")


def test_vision_mot_zoo_lists_80_families_3_variants() -> None:
    from dlhub.vision.mot_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 240
    assert "mot2d:sort_tiny" in arches
    assert "mot2d:deepsort_small" in arches
    assert "mot2d:bytetrack_base" in arches
    assert "mot2d:fairmot_tiny" in arches
    assert "mot2d:trackformer_small" in arches
    assert "mot2d:pmbm_gmphd_base" in arches
    assert "mot2d:motdt_tiny" in arches
    assert "mot2d:motip_base" in arches
    assert "mot2d:motrv2_tiny" in arches
    assert "mot2d:masktrack_rcnn_base" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "mot2d:sort_tiny",
        "mot2d:bytetrack_tiny",
        "mot2d:fairmot_tiny",
        "mot2d:transtrack_tiny",
    ],
)
def test_vision_mot_zoo_build_and_track_smoke(arch_id: str) -> None:
    from dlhub.vision.mot_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        num_classes=3,
        seq_len=4,
        image_size=64,
        width_mult=0.5,
        dropout=0.0,
    )
    x = torch.randn(2, 4, 3, 64, 64)
    out = model.track(x)
    assert isinstance(out, dict)
    assert "track_boxes" in out and "track_scores" in out and "track_ids" in out
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
