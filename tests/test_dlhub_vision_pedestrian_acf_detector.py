import pytest

torch = pytest.importorskip("torch")


def test_pedestrian_acf_detector_forward_backward_smoke() -> None:
    from dlhub.vision.detection._aliases import sum_output_means
    from dlhub.vision.detection_zoo import build_local_model, list_local_arches

    assert "dldet:pedestrian_acf" in list_local_arches()

    model = build_local_model("dldet:pedestrian_acf", in_channels=3, num_classes=1, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"score_map", "boxes"}
    assert out["score_map"].ndim == 4
    assert out["boxes"].ndim == 3
    assert out["boxes"].shape[-1] == 4

    loss = sum_output_means(out)
    assert torch.isfinite(loss)
    loss.backward()

