import pytest

torch = pytest.importorskip("torch")


def test_vision_synth_pedestrian_fcos_shapes_and_loss_smoke() -> None:
    from tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.model import (
        ModelConfig,
        build_model,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            image_size=64,
            stride=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.1,
        )
    )
    x, targets = next(iter(train_loader))
    assert tuple(x.shape) == (4, 3, 64, 64)
    assert set(targets.keys()) >= {"cls_target", "reg_target", "pos_mask", "box"}

    model = build_model(
        ModelConfig(arch="dldet:pedestrian_fcos", in_channels=3, num_classes=1, width_mult=0.5)
    )
    out = model(x)
    assert set(out.keys()) >= {"cls_logits", "reg"}

    cls_logits = out["cls_logits"]
    reg = out["reg"]
    cls_target = targets["cls_target"]
    reg_target = targets["reg_target"]
    pos_mask = targets["pos_mask"]

    assert tuple(cls_logits.shape) == tuple(cls_target.shape)
    assert tuple(reg.shape) == tuple(reg_target.shape)

    cls_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]))(cls_logits, cls_target)
    pred_pos = (reg * pos_mask).sum(dim=(2, 3))
    target_pos = (reg_target * pos_mask).sum(dim=(2, 3))
    reg_loss = torch.nn.SmoothL1Loss()(pred_pos, target_pos)

    loss = cls_loss + reg_loss
    assert torch.isfinite(loss)

