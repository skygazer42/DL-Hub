import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_medical_segmentation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_84_synthetic_medical_segmentation.data import (
        DataConfig,
        SyntheticMedicalSegmentationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_84_synthetic_medical_segmentation.model import (
        MedicalSegmentationModel,
        ModelConfig,
        medical_segmentation_loss,
        mean_dice,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=4,
        image_size=64,
        val_fraction=0.2,
        seed=7,
        num_workers=0,
        in_channels=1,
        num_classes=3,
    )
    ds = SyntheticMedicalSegmentationDataset(cfg)
    image, mask = ds[0]
    assert tuple(image.shape) == (1, 64, 64)
    assert tuple(mask.shape) == (64, 64)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.long
    assert 0.0 <= float(image.min().item()) <= float(image.max().item()) <= 1.0
    assert int(mask.min().item()) >= 0
    assert int(mask.max().item()) < 3

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_masks = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 1, 64, 64)
    assert tuple(batch_masks.shape) == (4, 64, 64)

    model = MedicalSegmentationModel(
        ModelConfig(
            in_channels=1,
            num_classes=3,
            backbone_family="unet",
            backbone_variant="unet_tiny",
        )
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {"logits", "mask"}
    assert tuple(outputs["logits"].shape) == (4, 3, 64, 64)
    assert tuple(outputs["mask"].shape) == (4, 64, 64)

    loss, parts = medical_segmentation_loss(outputs, batch_masks)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"cross_entropy"}
    assert float(parts["cross_entropy"]) >= 0.0
    dice = mean_dice(outputs["logits"], batch_masks, num_classes=3)
    assert 0.0 <= float(dice) <= 1.0
    loss.backward()


def test_vision_medical_segmentation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_84_synthetic_medical_segmentation.data import DataConfig
    from tracks.vision.lesson_84_synthetic_medical_segmentation.model import ModelConfig
    from tracks.vision.lesson_84_synthetic_medical_segmentation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_medical_segmentation_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=64,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=1,
            num_classes=3,
        ),
        ModelConfig(
            in_channels=1,
            num_classes=3,
            backbone_family="unet",
            backbone_variant="unet_tiny",
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_84_synthetic_medical_segmentation"
        / "pytest_medical_segmentation_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    record = metrics[0]
    for key in ("train_loss", "train_dice", "eval_loss", "eval_dice"):
        assert key in record
        assert float(record[key]) >= 0.0
