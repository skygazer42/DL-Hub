import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_text_clustering_batch_contract() -> None:
    from tracks.nlp.lesson_16_compact_text_clustering.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_16_compact_text_clustering.model import (
        ModelConfig,
        TextClusteringModel,
        cluster_accuracy,
        clustering_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=10,
            num_clusters=4,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "cluster_labels"}
    assert tuple(batch["input_ids"].shape) == (6, 10)
    assert tuple(batch["attention_mask"].shape) == (6, 10)
    assert tuple(batch["cluster_labels"].shape) == (6,)

    model = TextClusteringModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            proj_dim=24,
            num_clusters=4,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"embeddings", "logits"}
    assert tuple(outputs["embeddings"].shape) == (6, 24)
    assert tuple(outputs["logits"].shape) == (6, 4)

    loss = clustering_loss(outputs["logits"], batch["cluster_labels"])
    assert torch.isfinite(loss)
    acc = cluster_accuracy(outputs["logits"], batch["cluster_labels"])
    assert 0.0 <= acc <= 1.0


def test_text_clustering_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_16_compact_text_clustering.data import DataConfig
    from tracks.nlp.lesson_16_compact_text_clustering.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=7,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_text_clustering_smoke",
                embed_dim=32,
                proj_dim=24,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=10,
                num_clusters=4,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_16_compact_text_clustering" / "pytest_text_clustering_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_cluster_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_cluster_acc"] <= 1.0
