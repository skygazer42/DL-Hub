import pytest

torch = pytest.importorskip("torch")


def test_soft_nms_gaussian_decays_overlapping_scores() -> None:
    from dlhub.vision.detection._postprocess import soft_nms

    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])

    keep, scores_out = soft_nms(
        boxes,
        scores,
        iou_threshold=0.5,
        sigma=0.5,
        score_threshold=0.0,
        method="gaussian",
    )

    assert keep.dtype == torch.int64
    assert scores_out.shape == scores.shape
    assert set(keep.tolist()) == {0, 1, 2}
    assert keep[0].item() == 0

    # Overlapping box score should be decayed.
    assert float(scores_out[1]) < float(scores[1])
    # Non-overlapping box score should remain unchanged (within float tolerance).
    assert float(scores_out[2]) == pytest.approx(float(scores[2]), rel=0.0, abs=1e-6)


def test_diou_nms_suppresses_highly_overlapping_boxes() -> None:
    from dlhub.vision.detection._postprocess import diou_nms

    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])

    keep = diou_nms(boxes, scores, threshold=0.6)
    assert keep.dtype == torch.int64
    assert set(keep.tolist()) == {0, 2}

