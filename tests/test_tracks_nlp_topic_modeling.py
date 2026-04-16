import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_topic_modeling_batch_contract() -> None:
    from tracks.nlp.lesson_18_toy_topic_modeling.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_18_toy_topic_modeling.model import (
        ModelConfig,
        TopicModelingModel,
        topic_accuracy,
        topic_modeling_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=10,
            num_topics=4,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "bow_targets", "topic_labels"}
    assert tuple(batch["input_ids"].shape) == (6, 10)
    assert tuple(batch["attention_mask"].shape) == (6, 10)
    assert tuple(batch["bow_targets"].shape) == (6, vocab.size)
    assert tuple(batch["topic_labels"].shape) == (6,)

    model = TopicModelingModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            hidden_dim=24,
            num_topics=4,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"topic_probs", "reconstruction_logits"}
    assert tuple(outputs["topic_probs"].shape) == (6, 4)
    assert tuple(outputs["reconstruction_logits"].shape) == (6, vocab.size)

    loss, parts = topic_modeling_loss(
        outputs["reconstruction_logits"], batch["bow_targets"], outputs["topic_probs"]
    )
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"recon_loss", "entropy_loss"}
    assert float(parts["recon_loss"]) >= 0.0
    assert float(parts["entropy_loss"]) >= 0.0
    acc = topic_accuracy(outputs["topic_probs"], batch["topic_labels"])
    assert 0.0 <= acc <= 1.0


def test_topic_modeling_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_18_toy_topic_modeling.data import DataConfig
    from tracks.nlp.lesson_18_toy_topic_modeling.train import TrainConfig, run_training

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
                run_name="pytest_topic_modeling_smoke",
                embed_dim=32,
                hidden_dim=24,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=10,
                num_topics=4,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_18_toy_topic_modeling" / "pytest_topic_modeling_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert metric_row["train_recon_loss"] >= 0.0
    assert metric_row["eval_loss"] >= 0.0
    assert metric_row["eval_recon_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_topic_acc"] <= 1.0
