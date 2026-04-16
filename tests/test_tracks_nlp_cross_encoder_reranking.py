import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_cross_encoder_reranking_batch_contract() -> None:
    from tracks.nlp.lesson_15_toy_cross_encoder_reranking.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_15_toy_cross_encoder_reranking.model import (
        CrossEncoderReranker,
        ModelConfig,
        reranking_accuracy,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_query_length=8,
            max_doc_length=10,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {
        "positive_input_ids",
        "positive_attention_mask",
        "negative_input_ids",
        "negative_attention_mask",
    }
    assert tuple(batch["positive_input_ids"].shape) == (6, 19)
    assert tuple(batch["negative_input_ids"].shape) == (6, 19)

    model = CrossEncoderReranker(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=19,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            ff_dim=64,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"positive_scores", "negative_scores"}
    assert tuple(outputs["positive_scores"].shape) == (6,)
    assert tuple(outputs["negative_scores"].shape) == (6,)

    loss = torch.nn.functional.softplus(-(outputs["positive_scores"] - outputs["negative_scores"])).mean()
    assert torch.isfinite(loss)
    acc = reranking_accuracy(outputs["positive_scores"], outputs["negative_scores"])
    assert 0.0 <= acc <= 1.0


def test_cross_encoder_reranking_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_15_toy_cross_encoder_reranking.data import DataConfig
    from tracks.nlp.lesson_15_toy_cross_encoder_reranking.train import TrainConfig, run_training

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
                run_name="pytest_cross_encoder_reranking_smoke",
                embed_dim=32,
                num_heads=4,
                num_layers=1,
                ff_dim=64,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_query_length=8,
                max_doc_length=10,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_15_toy_cross_encoder_reranking" / "pytest_cross_encoder_reranking_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_rerank_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_rerank_acc"] <= 1.0
