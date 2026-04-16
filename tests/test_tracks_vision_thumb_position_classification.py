import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_thumb_position_classification_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_57_synthetic_thumb_position_classification.data import (
        DataConfig,
        SyntheticThumbPositionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_57_synthetic_thumb_position_classification.model import (
        ModelConfig,
        ThumbPositionClassifier,
        thumb_position_accuracy,
        thumb_position_loss,
    )

    cfg = DataConfig(
        num_samples=42,
        batch_size=6,
        image_size=64,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        in_channels=1,
        num_classes=3,
    )

    dataset = SyntheticThumbPositionDataset(cfg)
    image, label = dataset[0]
    assert tuple(image.shape) == (1, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 3
    assert image.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))
    assert tuple(images.shape) == (6, 1, 64, 64)
    assert tuple(labels.shape) == (6,)
    assert labels.dtype == torch.int64
    assert torch.all(labels >= 0)
    assert torch.all(labels < 3)

    model = ThumbPositionClassifier(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_classes=3, dropout=0.0)
    )
    logits = model(images)
    assert tuple(logits.shape) == (6, 3)

    loss = thumb_position_loss(logits, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= thumb_position_accuracy(logits.detach(), labels) <= 1.0
    loss.backward()


def test_thumb_position_classification_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_57_synthetic_thumb_position_classification.data import DataConfig
    from tracks.vision.lesson_57_synthetic_thumb_position_classification.model import ModelConfig
    from tracks.vision.lesson_57_synthetic_thumb_position_classification.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=57,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_thumb_position_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=64,
            val_fraction=0.2,
            seed=7,
            num_workers=0,
            in_channels=1,
            num_classes=3,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_classes=3, dropout=0.0),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_57_synthetic_thumb_position_classification"
        / "pytest_thumb_position_smoke"
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
    for key in ("train_loss", "train_acc", "eval_loss", "eval_acc"):
        assert key in record
    assert float(record["train_loss"]) >= 0.0
    assert float(record["eval_loss"]) >= 0.0
    assert 0.0 <= float(record["train_acc"]) <= 1.0
    assert 0.0 <= float(record["eval_acc"]) <= 1.0
