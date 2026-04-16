import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_meta_few_shot_text_episode_shapes() -> None:
    from tracks.nlp.lesson_24_toy_meta_few_shot_text_classification.data import (
        DataConfig,
        get_dataloaders,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_episodes=24,
            batch_size=3,
            num_ways=4,
            shots=2,
            queries_per_class=2,
            max_length=12,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert batch["support_input_ids"].shape == (3, 8, 12)
    assert batch["support_attention_mask"].shape == (3, 8, 12)
    assert batch["support_labels"].shape == (3, 8)
    assert batch["query_input_ids"].shape == (3, 8, 12)
    assert batch["query_attention_mask"].shape == (3, 8, 12)
    assert batch["query_labels"].shape == (3, 8)
    assert len(batch["task_names"]) == 3
    assert "adapt" in vocab.token_to_id


def test_meta_few_shot_text_model_forward_and_metrics() -> None:
    from tracks.nlp.lesson_24_toy_meta_few_shot_text_classification.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_24_toy_meta_few_shot_text_classification.model import (
        MetaFewShotTextClassifier,
        ModelConfig,
        episode_accuracy,
        meta_episode_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_episodes=18,
            batch_size=2,
            num_ways=3,
            shots=2,
            queries_per_class=2,
            max_length=10,
            val_fraction=0.25,
            seed=1,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))
    model = MetaFewShotTextClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            proj_dim=24,
            dropout=0.0,
        )
    )

    outputs = model(batch)
    assert set(outputs) == {"support_embeddings", "query_embeddings", "prototypes", "logits"}
    assert outputs["support_embeddings"].shape == (2, 6, 24)
    assert outputs["query_embeddings"].shape == (2, 6, 24)
    assert outputs["prototypes"].shape == (2, 3, 24)
    assert outputs["logits"].shape == (2, 6, 3)

    loss = meta_episode_loss(outputs["logits"], batch["query_labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    acc = episode_accuracy(outputs["logits"], batch["query_labels"])
    assert 0.0 <= acc <= 1.0


def test_meta_few_shot_text_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_24_toy_meta_few_shot_text_classification.data import DataConfig
    from tracks.nlp.lesson_24_toy_meta_few_shot_text_classification.train import (
        TrainConfig,
        run_training,
    )

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
                run_name="pytest_meta_few_shot_smoke",
                embed_dim=32,
                proj_dim=24,
                dropout=0.0,
            ),
            DataConfig(
                num_episodes=24,
                batch_size=3,
                num_ways=3,
                shots=2,
                queries_per_class=2,
                max_length=12,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_24_toy_meta_few_shot_text_classification" / "pytest_meta_few_shot_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_episode_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_episode_acc"] <= 1.0
