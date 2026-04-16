import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_co_segmentation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_86_synthetic_co_segmentation.data import (
        DataConfig,
        SyntheticCoSegmentationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_86_synthetic_co_segmentation.model import (
        CoSegmentationModel,
        ModelConfig,
        co_segmentation_loss,
        mask_iou,
    )

    cfg = DataConfig(
        num_samples=24,
        batch_size=2,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        set_size=3,
    )
    ds = SyntheticCoSegmentationDataset(cfg)
    images, targets = ds[0]

    assert tuple(images.shape) == (3, 3, 32, 32)
    assert tuple(targets["mask"].shape) == (3, 32, 32)
    assert tuple(targets["class_index"].shape) == (3, 32, 32)
    assert images.dtype == torch.float32
    assert targets["mask"].dtype == torch.float32
    assert targets["class_index"].dtype == torch.long

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (2, 3, 3, 32, 32)
    assert tuple(batch_targets["mask"].shape) == (2, 3, 32, 32)
    assert tuple(batch_targets["class_index"].shape) == (2, 3, 32, 32)

    model = CoSegmentationModel(
        ModelConfig(in_channels=3, num_classes=2, arch="coseg:siamese_coseg_tiny")
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {"logits", "mask", "group_tokens", "match_map"}
    assert tuple(outputs["logits"].shape) == (2, 3, 2, 32, 32)
    assert tuple(outputs["mask"].shape) == (2, 3, 32, 32)
    assert tuple(outputs["group_tokens"].shape) == (2, 3, 64)
    assert tuple(outputs["match_map"].shape) == (2, 3, 1, 32, 32)

    loss, parts = co_segmentation_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"cross_entropy", "dice_loss"}
    assert float(parts["cross_entropy"]) >= 0.0
    assert float(parts["dice_loss"]) >= 0.0
    assert 0.0 <= mask_iou(outputs["mask"], batch_targets["mask"]) <= 1.0
    loss.backward()


def test_vision_co_segmentation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_86_synthetic_co_segmentation.data import DataConfig
    from tracks.vision.lesson_86_synthetic_co_segmentation.model import ModelConfig
    from tracks.vision.lesson_86_synthetic_co_segmentation.train import (
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
            run_name="pytest_co_segmentation_smoke",
        ),
        DataConfig(
            num_samples=24,
            batch_size=2,
            image_size=32,
            val_fraction=0.2,
            seed=7,
            num_workers=0,
            in_channels=3,
            set_size=3,
        ),
        ModelConfig(in_channels=3, num_classes=2, arch="coseg:siamese_coseg_tiny"),
    )
    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_86_synthetic_co_segmentation" / "pytest_co_segmentation_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    row = records[0]
    for key in (
        "train_loss",
        "train_cross_entropy",
        "train_dice_loss",
        "train_iou",
        "eval_loss",
        "eval_cross_entropy",
        "eval_dice_loss",
        "eval_iou",
    ):
        assert key in row
        assert float(row[key]) >= 0.0
