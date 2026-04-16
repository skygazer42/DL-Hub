import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_face_retrieval_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_46_synthetic_face_retrieval.data import (
        DataConfig,
        SyntheticFaceRetrievalDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_46_synthetic_face_retrieval.model import (
        FaceRetrievalEmbeddingNet,
        ModelConfig,
        retrieval_top1_accuracy,
        triplet_margin_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=5,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
        num_identities=5,
    )
    ds = SyntheticFaceRetrievalDataset(cfg)
    image, label = ds[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert 0 <= label < 5

    train_loader, _ = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))
    assert tuple(images.shape) == (5, 1, 48, 48)
    assert tuple(labels.shape) == (5,)

    model = FaceRetrievalEmbeddingNet(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, embedding_dim=16, dropout=0.0)
    )
    embeddings = model(images)
    assert tuple(embeddings.shape) == (5, 16)

    loss = triplet_margin_loss(embeddings, labels, margin=0.2)
    assert torch.isfinite(loss)
    assert 0.0 <= retrieval_top1_accuracy(embeddings.detach(), labels) <= 1.0
    loss.backward()


def test_face_retrieval_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_46_synthetic_face_retrieval.data import DataConfig
    from tracks.vision.lesson_46_synthetic_face_retrieval.model import ModelConfig
    from tracks.vision.lesson_46_synthetic_face_retrieval.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_face_retrieval_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
            num_identities=5,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, embedding_dim=16, dropout=0.1),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_46_synthetic_face_retrieval" / "pytest_face_retrieval_smoke"
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
