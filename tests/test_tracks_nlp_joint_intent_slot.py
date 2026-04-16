import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_joint_intent_slot_batch_contract() -> None:
    from tracks.nlp.lesson_26_toy_joint_intent_slot_parsing.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_26_toy_joint_intent_slot_parsing.model import (
        JointIntentSlotModel,
        ModelConfig,
        compute_joint_metrics,
        joint_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=60,
            batch_size=5,
            max_length=10,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "intent_labels", "slot_labels"}
    assert tuple(batch["input_ids"].shape) == (5, 10)
    assert tuple(batch["attention_mask"].shape) == (5, 10)
    assert tuple(batch["intent_labels"].shape) == (5,)
    assert tuple(batch["slot_labels"].shape) == (5, 10)
    assert "book" in vocab.token_to_id
    assert "city" in vocab.token_to_id

    model = JointIntentSlotModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_intents=4,
            num_slot_labels=7,
            embed_dim=32,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"intent_logits", "slot_logits"}
    assert tuple(outputs["intent_logits"].shape) == (5, 4)
    assert tuple(outputs["slot_logits"].shape) == (5, 10, 7)

    loss = joint_loss(
        outputs["intent_logits"],
        outputs["slot_logits"],
        batch["intent_labels"],
        batch["slot_labels"],
        batch["attention_mask"],
    )
    assert torch.isfinite(loss)

    metrics = compute_joint_metrics(
        outputs["intent_logits"],
        outputs["slot_logits"],
        batch["intent_labels"],
        batch["slot_labels"],
        batch["attention_mask"],
    )
    assert 0.0 <= metrics["intent_acc"] <= 1.0
    assert 0.0 <= metrics["slot_token_acc"] <= 1.0


def test_joint_intent_slot_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_26_toy_joint_intent_slot_parsing.data import DataConfig
    from tracks.nlp.lesson_26_toy_joint_intent_slot_parsing.train import TrainConfig, run_training

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
                run_name="pytest_joint_intent_slot_smoke",
                embed_dim=32,
                dropout=0.0,
                slot_loss_weight=1.0,
            ),
            DataConfig(
                num_samples=80,
                batch_size=8,
                max_length=10,
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
        / "lesson_26_toy_joint_intent_slot_parsing"
        / "pytest_joint_intent_slot_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_intent_acc"] <= 1.0
    assert 0.0 <= metric_row["train_slot_token_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_intent_acc"] <= 1.0
    assert 0.0 <= metric_row["eval_slot_token_acc"] <= 1.0
