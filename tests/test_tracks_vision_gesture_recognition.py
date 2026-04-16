import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_gesture_recognition_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_51_synthetic_gesture_recognition.data import (
        DataConfig,
        SyntheticGestureRecognitionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_51_synthetic_gesture_recognition.model import (
        GestureRecognitionClassifier,
        ModelConfig,
        gesture_accuracy,
        gesture_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=5,
        image_size=64,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        in_channels=1,
        num_classes=4,
    )

    dataset = SyntheticGestureRecognitionDataset(cfg)
    image, label = dataset[0]
    assert tuple(image.shape) == (1, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 4
    assert image.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))
    assert tuple(images.shape) == (5, 1, 64, 64)
    assert tuple(labels.shape) == (5,)
    assert labels.dtype == torch.int64

    model = GestureRecognitionClassifier(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_classes=4, dropout=0.0)
    )
    logits = model(images)
    assert tuple(logits.shape) == (5, 4)

    loss = gesture_loss(logits, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= gesture_accuracy(logits.detach(), labels) <= 1.0
    loss.backward()


def test_gesture_recognition_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_51_synthetic_gesture_recognition.data import DataConfig
    from tracks.vision.lesson_51_synthetic_gesture_recognition.model import ModelConfig
    from tracks.vision.lesson_51_synthetic_gesture_recognition.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_gesture_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=64,
            val_fraction=0.2,
            seed=7,
            num_workers=0,
            in_channels=1,
            num_classes=4,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_classes=4, dropout=0.0),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_51_synthetic_gesture_recognition"
        / "pytest_gesture_smoke"
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
