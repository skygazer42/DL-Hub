import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_fine_grained_recognition_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_64_synthetic_fine_grained_recognition.data import (
        DataConfig,
        SyntheticFineGrainedRecognitionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_64_synthetic_fine_grained_recognition.model import (
        FineGrainedRecognitionClassifier,
        ModelConfig,
        fine_grained_accuracy,
        fine_grained_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=5,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
        num_classes=6,
    )
    dataset = SyntheticFineGrainedRecognitionDataset(cfg)
    image, label = dataset[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert 0 <= label < 6

    train_loader, _ = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))
    assert tuple(images.shape) == (5, 1, 48, 48)
    assert tuple(labels.shape) == (5,)

    model = FineGrainedRecognitionClassifier(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_classes=6, dropout=0.0)
    )
    logits = model(images)
    assert tuple(logits.shape) == (5, 6)

    loss = fine_grained_loss(logits, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= fine_grained_accuracy(logits.detach(), labels) <= 1.0
    loss.backward()


def test_fine_grained_recognition_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_64_synthetic_fine_grained_recognition.data import DataConfig
    from tracks.vision.lesson_64_synthetic_fine_grained_recognition.model import ModelConfig
    from tracks.vision.lesson_64_synthetic_fine_grained_recognition.train import (
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
            run_name="pytest_fine_grained_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
            num_classes=6,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_classes=6, dropout=0.1),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_64_synthetic_fine_grained_recognition"
        / "pytest_fine_grained_smoke"
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
        assert float(record[key]) >= 0.0
