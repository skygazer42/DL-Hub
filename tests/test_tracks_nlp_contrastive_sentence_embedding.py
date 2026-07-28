import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_contrastive_sentence_embedding_batch_contract() -> None:
    from tracks.nlp.lesson_14_compact_contrastive_sentence_embedding.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_14_compact_contrastive_sentence_embedding.model import (
        ContrastiveSentenceEncoder,
        ModelConfig,
        contrastive_accuracy,
        nt_xent_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=10,
            dropout_prob=0.2,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs = next(iter(train_loader))

    assert set(inputs.keys()) == {
        "view1_input_ids",
        "view1_attention_mask",
        "view2_input_ids",
        "view2_attention_mask",
    }
    assert tuple(inputs["view1_input_ids"].shape) == (6, 10)
    assert tuple(inputs["view2_input_ids"].shape) == (6, 10)

    model = ContrastiveSentenceEncoder(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            proj_dim=24,
            dropout=0.0,
        )
    )
    outputs = model(inputs)
    assert set(outputs.keys()) == {"view1_embeddings", "view2_embeddings", "sim_matrix"}
    assert tuple(outputs["view1_embeddings"].shape) == (6, 24)
    assert tuple(outputs["view2_embeddings"].shape) == (6, 24)
    assert tuple(outputs["sim_matrix"].shape) == (6, 6)

    loss = nt_xent_loss(outputs["sim_matrix"])
    assert torch.isfinite(loss)
    acc = contrastive_accuracy(outputs["sim_matrix"])
    assert 0.0 <= acc <= 1.0


def test_contrastive_sentence_embedding_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_14_compact_contrastive_sentence_embedding.data import DataConfig
    from tracks.nlp.lesson_14_compact_contrastive_sentence_embedding.train import TrainConfig, run_training

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
                run_name="pytest_contrastive_sentence_smoke",
                embed_dim=32,
                proj_dim=24,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=10,
                dropout_prob=0.2,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_14_compact_contrastive_sentence_embedding" / "pytest_contrastive_sentence_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_contrastive_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_contrastive_acc"] <= 1.0


