import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_text_recognition_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_36_synthetic_text_recognition.data import (
        DataConfig,
        SyntheticWordDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_36_synthetic_text_recognition.model import (
        ModelConfig,
        TextRecognizer,
        sequence_accuracy,
        sequence_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_height=24,
        image_width=80,
        text_length=5,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        in_channels=1,
    )
    dataset = SyntheticWordDataset(cfg)
    image, label = dataset[0]
    assert tuple(image.shape) == (1, 24, 80)
    assert tuple(label.shape) == (5,)

    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 24, 80)
    assert tuple(labels.shape) == (4, 5)

    model = TextRecognizer(
        ModelConfig(
            vocab_size=vocab.size,
            in_channels=1,
            text_length=5,
            hidden_channels=20,
            dropout=0.0,
        )
    )
    logits = model(images)
    assert tuple(logits.shape) == (4, 5, vocab.size)

    loss = sequence_loss(logits, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= sequence_accuracy(logits.detach(), labels) <= 1.0
    loss.backward()


def test_vision_text_recognition_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_36_synthetic_text_recognition.data import DataConfig
    from tracks.vision.lesson_36_synthetic_text_recognition.model import ModelConfig
    from tracks.vision.lesson_36_synthetic_text_recognition.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_text_recognition_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_height=24,
            image_width=80,
            text_length=5,
            val_fraction=0.25,
            seed=5,
            num_workers=0,
            in_channels=1,
        ),
        ModelConfig(vocab_size=37, in_channels=1, text_length=5, hidden_channels=20, dropout=0.1),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_36_synthetic_text_recognition" / "pytest_text_recognition_smoke"
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
    for key in ("train_loss", "train_seq_acc", "eval_loss", "eval_seq_acc"):
        assert key in record
        assert float(record[key]) >= 0.0
