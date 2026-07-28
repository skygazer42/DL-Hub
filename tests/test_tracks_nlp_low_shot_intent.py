import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_low_shot_intent_batch_contract() -> None:
    from tracks.nlp.lesson_25_compact_low_shot_intent_detection.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_25_compact_low_shot_intent_detection.model import (
        LowShotIntentClassifier,
        ModelConfig,
        classification_accuracy,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=12,
            num_classes=4,
            shots_per_class=3,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert tuple(batch["input_ids"].shape) == (6, 12)
    assert tuple(batch["attention_mask"].shape) == (6, 12)
    assert tuple(batch["labels"].shape) == (6,)
    assert "intent" in vocab.token_to_id
    assert "fewshot" in vocab.token_to_id

    model = LowShotIntentClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            num_classes=4,
            dropout=0.0,
        )
    )
    logits = model(batch)
    assert tuple(logits.shape) == (6, 4)
    loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
    assert torch.isfinite(loss)
    assert 0.0 <= classification_accuracy(logits.detach(), batch["labels"]) <= 1.0


def test_low_shot_intent_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_25_compact_low_shot_intent_detection.data import DataConfig
    from tracks.nlp.lesson_25_compact_low_shot_intent_detection.train import TrainConfig, run_training

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
                run_name="pytest_low_shot_intent_smoke",
                embed_dim=32,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=12,
                num_classes=4,
                shots_per_class=3,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_25_compact_low_shot_intent_detection" / "pytest_low_shot_intent_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_acc"] <= 1.0
