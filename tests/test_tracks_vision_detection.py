import pytest


torch = pytest.importorskip("torch")


def test_vision_synth_fcos_shapes_and_loss_smoke() -> None:
    from tracks.vision.lesson_04_synthetic_detection_fcos.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_04_synthetic_detection_fcos.model import ModelConfig, TinyFCOS

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=4,
            image_size=64,
            stride=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.1,
            min_box_size=10,
            max_box_size=20,
        )
    )
    x, targets = next(iter(train_loader))

    assert tuple(x.shape) == (4, 1, 64, 64)
    assert set(targets.keys()) == {"cls_target", "reg_target", "pos_mask", "box"}

    model = TinyFCOS(ModelConfig(in_channels=1, hidden_channels=32, stride=4))
    out = model(x)
    assert set(out.keys()) == {"cls_logits", "reg"}
    assert tuple(out["cls_logits"].shape) == (4, 1, 16, 16)
    assert tuple(out["reg"].shape) == (4, 4, 16, 16)

    cls_target = targets["cls_target"]
    reg_target = targets["reg_target"]
    pos_mask = targets["pos_mask"]

    cls_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]))(out["cls_logits"], cls_target)
    pred_pos = (out["reg"] * pos_mask).sum(dim=(2, 3))
    target_pos = (reg_target * pos_mask).sum(dim=(2, 3))
    reg_loss = torch.nn.SmoothL1Loss()(pred_pos, target_pos)

    loss = cls_loss + 2.0 * reg_loss
    assert torch.isfinite(loss)

