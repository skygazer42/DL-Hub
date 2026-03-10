import pytest

torch = pytest.importorskip("torch")


def test_vision_synth_instance_seg_shapes_and_one_step_smoke() -> None:
    from tracks.vision.lesson_11_synthetic_instance_segmentation_yolact.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.vision.lesson_11_synthetic_instance_segmentation_yolact.model import (
        ModelConfig,
        TinyYOLACT,
        mask_logits_from_proto,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=2,
            image_size=64,
            stride=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.1,
            min_rect=10,
            max_rect=20,
        )
    )
    x, targets = next(iter(train_loader))

    assert tuple(x.shape) == (2, 1, 64, 64)
    assert set(targets.keys()) == {"cls_target", "reg_target", "pos_mask", "mask", "box"}
    assert tuple(targets["cls_target"].shape) == (2, 1, 8, 8)
    assert tuple(targets["reg_target"].shape) == (2, 4, 8, 8)
    assert tuple(targets["pos_mask"].shape) == (2, 1, 8, 8)
    assert tuple(targets["mask"].shape) == (2, 1, 64, 64)

    model = TinyYOLACT(
        ModelConfig(in_channels=1, num_classes=1, variant="yolact_tiny", width_mult=0.5)
    )
    out = model(x)
    assert set(out.keys()) == {"proto", "cls_logits", "bbox_deltas", "mask_coeffs"}

    proto = out["proto"]
    cls_logits = out["cls_logits"]
    bbox = out["bbox_deltas"]
    coeffs = out["mask_coeffs"]
    assert proto.ndim == 4 and cls_logits.ndim == 4 and bbox.ndim == 4 and coeffs.ndim == 4
    assert (
        proto.shape[0] == 2
        and cls_logits.shape[0] == 2
        and bbox.shape[0] == 2
        and coeffs.shape[0] == 2
    )
    assert tuple(cls_logits.shape[1:]) == (1, 8, 8)
    assert tuple(bbox.shape[1:]) == (4, 8, 8)
    assert coeffs.shape[2:] == (8, 8)
    assert proto.shape[2:] == (16, 16)
    assert coeffs.shape[1] == proto.shape[1]

    pos_mask = targets["pos_mask"]
    mask_logits = mask_logits_from_proto(
        proto=proto,
        mask_coeffs=coeffs,
        pos_mask=pos_mask,
        out_hw=(64, 64),
    )
    assert tuple(mask_logits.shape) == (2, 1, 64, 64)

    cls_target = targets["cls_target"]
    reg_target = targets["reg_target"]
    mask_target = targets["mask"]

    cls_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]))(cls_logits, cls_target)
    pred_pos = (bbox * pos_mask).sum(dim=(2, 3))
    target_pos = (reg_target * pos_mask).sum(dim=(2, 3))
    reg_loss = torch.nn.SmoothL1Loss()(pred_pos, target_pos)
    mask_loss = torch.nn.BCEWithLogitsLoss()(mask_logits, mask_target)

    loss = cls_loss + 2.0 * reg_loss + 1.0 * mask_loss
    assert torch.isfinite(loss)
    loss.backward()
