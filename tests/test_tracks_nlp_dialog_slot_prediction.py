import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_dialog_slot_prediction_batch_contract() -> None:
    from tracks.nlp.lesson_36_toy_dialog_slot_prediction.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_36_toy_dialog_slot_prediction.model import (
        DialogSlotPredictor,
        ModelConfig,
        compute_slot_metrics,
        dialog_slot_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=72,
            batch_size=6,
            max_length=28,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "cuisine_labels",
        "area_labels",
        "party_labels",
    }
    assert tuple(batch["input_ids"].shape) == (6, 28)
    assert tuple(batch["attention_mask"].shape) == (6, 28)
    assert tuple(batch["cuisine_labels"].shape) == (6,)
    assert tuple(batch["area_labels"].shape) == (6,)
    assert tuple(batch["party_labels"].shape) == (6,)
    assert torch.all(batch["cuisine_labels"] >= 0)
    assert torch.all(batch["cuisine_labels"] < 4)
    assert torch.all(batch["area_labels"] >= 0)
    assert torch.all(batch["area_labels"] < 4)
    assert torch.all(batch["party_labels"] >= 0)
    assert torch.all(batch["party_labels"] < 4)
    assert "slot" in vocab.token_to_id
    assert "cuisine" in vocab.token_to_id
    assert "downtown" in vocab.token_to_id

    model = DialogSlotPredictor(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_cuisine_slots=4,
            num_area_slots=4,
            num_party_slots=4,
            embed_dim=32,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"cuisine_logits", "area_logits", "party_logits"}
    assert tuple(outputs["cuisine_logits"].shape) == (6, 4)
    assert tuple(outputs["area_logits"].shape) == (6, 4)
    assert tuple(outputs["party_logits"].shape) == (6, 4)

    loss = dialog_slot_loss(
        outputs["cuisine_logits"],
        outputs["area_logits"],
        outputs["party_logits"],
        batch["cuisine_labels"],
        batch["area_labels"],
        batch["party_labels"],
    )
    assert torch.isfinite(loss)

    metrics = compute_slot_metrics(
        outputs["cuisine_logits"],
        outputs["area_logits"],
        outputs["party_logits"],
        batch["cuisine_labels"],
        batch["area_labels"],
        batch["party_labels"],
    )
    assert 0.0 <= metrics["slot_acc"] <= 1.0
    assert 0.0 <= metrics["joint_goal_acc"] <= 1.0


def test_dialog_slot_prediction_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_36_toy_dialog_slot_prediction.data import DataConfig
    from tracks.nlp.lesson_36_toy_dialog_slot_prediction.train import TrainConfig, run_training

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
                run_name="pytest_dialog_slot_prediction_smoke",
                embed_dim=32,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=96,
                batch_size=8,
                max_length=28,
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
        / "lesson_36_toy_dialog_slot_prediction"
        / "pytest_dialog_slot_prediction_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_slot_acc"] <= 1.0
    assert 0.0 <= metric_row["train_joint_goal_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_slot_acc"] <= 1.0
    assert 0.0 <= metric_row["eval_joint_goal_acc"] <= 1.0
