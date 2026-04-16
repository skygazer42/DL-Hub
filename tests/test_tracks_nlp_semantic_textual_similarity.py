import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_semantic_textual_similarity_batch_contract() -> None:
    from tracks.nlp.lesson_28_toy_semantic_textual_similarity.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_28_toy_semantic_textual_similarity.model import (
        ModelConfig,
        SemanticTextualSimilarityRegressor,
        mean_absolute_error,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            max_length=16,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "scores"}
    assert tuple(batch["input_ids"].shape) == (8, 16)
    assert tuple(batch["attention_mask"].shape) == (8, 16)
    assert tuple(batch["scores"].shape) == (8,)
    assert batch["scores"].dtype == torch.float32
    assert torch.all(batch["scores"] >= 0.0)
    assert torch.all(batch["scores"] <= 1.0)
    assert "sentence" in vocab.token_to_id
    assert "a" in vocab.token_to_id
    assert "b" in vocab.token_to_id

    model = SemanticTextualSimilarityRegressor(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            hidden_dim=48,
            dropout=0.0,
        )
    )
    predictions = model(batch)
    assert tuple(predictions.shape) == (8,)
    loss = torch.nn.functional.mse_loss(predictions, batch["scores"])
    assert torch.isfinite(loss)
    assert mean_absolute_error(predictions.detach(), batch["scores"]) >= 0.0


def test_semantic_textual_similarity_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_28_toy_semantic_textual_similarity.data import DataConfig
    from tracks.nlp.lesson_28_toy_semantic_textual_similarity.train import TrainConfig, run_training

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
                run_name="pytest_semantic_textual_similarity_smoke",
                embed_dim=32,
                hidden_dim=48,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=96,
                batch_size=8,
                max_length=16,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = (
        tmp_path
        / "outputs"
        / "nlp"
        / "lesson_28_toy_semantic_textual_similarity"
        / "pytest_semantic_textual_similarity_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert metric_row["eval_loss"] >= 0.0
    assert metric_row["train_mae"] >= 0.0
    assert metric_row["eval_mae"] >= 0.0
