import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_face_verification_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_44_synthetic_face_verification.data import (
        DataConfig,
        SyntheticFaceVerificationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_44_synthetic_face_verification.model import (
        FaceVerificationModel,
        ModelConfig,
        verification_accuracy,
        verification_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=5,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
    )

    dataset = SyntheticFaceVerificationDataset(cfg)
    image_a, image_b, label = dataset[0]
    assert tuple(image_a.shape) == (1, 48, 48)
    assert tuple(image_b.shape) == (1, 48, 48)
    assert label in (0, 1)
    assert image_a.dtype == torch.float32
    assert image_b.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    images_a, images_b, labels = next(iter(train_loader))
    assert tuple(images_a.shape) == (5, 1, 48, 48)
    assert tuple(images_b.shape) == (5, 1, 48, 48)
    assert tuple(labels.shape) == (5,)
    assert labels.dtype == torch.int64

    model = FaceVerificationModel(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, embedding_dim=48, dropout=0.0)
    )
    logits = model(images_a, images_b)
    assert tuple(logits.shape) == (5, 2)

    loss = verification_loss(logits, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= verification_accuracy(logits.detach(), labels) <= 1.0
    loss.backward()


def test_face_verification_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_44_synthetic_face_verification.data import DataConfig
    from tracks.vision.lesson_44_synthetic_face_verification.model import ModelConfig
    from tracks.vision.lesson_44_synthetic_face_verification.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_face_verification_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, embedding_dim=64, dropout=0.1),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_44_synthetic_face_verification"
        / "pytest_face_verification_smoke"
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
