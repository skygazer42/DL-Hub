import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_reid_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_88_synthetic_reid.data import (
        DataConfig,
        SyntheticReIDDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_88_synthetic_reid.model import (
        ModelConfig,
        build_model,
        reid_loss,
        retrieval_top1_accuracy,
    )

    data_cfg = DataConfig(
        num_samples=48,
        batch_size=6,
        image_size=64,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        in_channels=3,
        num_identities=6,
        noise_std=0.02,
    )
    dataset = SyntheticReIDDataset(data_cfg)
    image, label = dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < 6

    train_loader, _ = get_dataloaders(data_cfg)
    images, labels = next(iter(train_loader))
    assert tuple(images.shape) == (6, 3, 64, 64)
    assert tuple(labels.shape) == (6,)
    assert labels.dtype == torch.long

    model_cfg = ModelConfig(
        in_channels=3,
        num_classes=6,
        arch="osnet:osnet_tiny",
        variant="",
        width_mult=1.0,
        dropout=0.0,
    )
    model = build_model(model_cfg)
    outputs = model(images)
    assert set(outputs.keys()) == {"embedding", "logits"}
    assert tuple(outputs["embedding"].shape) == (6, 96)
    assert tuple(outputs["logits"].shape) == (6, 6)

    loss, parts = reid_loss(outputs, labels)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"ce", "triplet"}
    assert float(parts["ce"]) >= 0.0
    assert float(parts["triplet"]) >= 0.0
    assert 0.0 <= retrieval_top1_accuracy(outputs["embedding"].detach(), labels) <= 1.0
    loss.backward()


def test_reid_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_88_synthetic_reid.data import DataConfig
    from tracks.vision.lesson_88_synthetic_reid.model import ModelConfig
    from tracks.vision.lesson_88_synthetic_reid.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_reid_smoke",
            arch="osnet:osnet_tiny",
            width_mult=1.0,
            dropout=0.1,
        ),
        DataConfig(
            num_samples=56,
            batch_size=7,
            image_size=64,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            in_channels=3,
            num_identities=7,
            noise_std=0.02,
        ),
        ModelConfig(
            in_channels=3,
            num_classes=7,
            arch="osnet:osnet_tiny",
            variant="",
            width_mult=1.0,
            dropout=0.1,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_88_synthetic_reid" / "pytest_reid_smoke"
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
    for key in ("train_loss", "train_top1", "eval_loss", "eval_top1"):
        assert key in record
        assert float(record[key]) >= 0.0
