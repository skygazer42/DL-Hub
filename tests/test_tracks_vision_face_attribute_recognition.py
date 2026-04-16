import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_face_attribute_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_40_synthetic_face_attribute_recognition.data import (
        DataConfig,
        SyntheticFaceAttributeDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_40_synthetic_face_attribute_recognition.model import (
        FaceAttributeClassifier,
        ModelConfig,
        attribute_accuracy,
        attribute_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        num_attributes=3,
    )

    dataset = SyntheticFaceAttributeDataset(cfg)
    image, attributes = dataset[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert tuple(attributes.shape) == (3,)
    assert image.dtype == torch.float32
    assert attributes.dtype == torch.float32
    assert torch.all(attributes >= 0.0)
    assert torch.all(attributes <= 1.0)

    train_loader, _ = get_dataloaders(cfg)
    images, targets = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 48, 48)
    assert tuple(targets.shape) == (4, 3)

    model = FaceAttributeClassifier(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_attributes=3, dropout=0.0)
    )
    logits = model(images)
    assert tuple(logits.shape) == (4, 3)

    loss = attribute_loss(logits, targets)
    assert torch.isfinite(loss)
    assert 0.0 <= attribute_accuracy(logits.detach(), targets) <= 1.0
    loss.backward()


def test_face_attribute_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_40_synthetic_face_attribute_recognition.data import DataConfig
    from tracks.vision.lesson_40_synthetic_face_attribute_recognition.model import ModelConfig
    from tracks.vision.lesson_40_synthetic_face_attribute_recognition.train import (
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
            run_name="pytest_face_attribute_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            num_attributes=3,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_attributes=3, dropout=0.0),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_40_synthetic_face_attribute_recognition"
        / "pytest_face_attribute_smoke"
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
    for key in ("train_bce", "eval_bce", "eval_attr_acc"):
        assert key in record
    assert float(record["train_bce"]) >= 0.0
    assert float(record["eval_bce"]) >= 0.0
    assert 0.0 <= float(record["eval_attr_acc"]) <= 1.0
