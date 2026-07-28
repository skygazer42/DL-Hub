import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_dialog_state_tracking_batch_contract() -> None:
    from tracks.nlp.lesson_29_compact_dialog_state_tracking.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_29_compact_dialog_state_tracking.model import (
        DialogStateTracker,
        ModelConfig,
        compute_state_metrics,
        dialog_state_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=60,
            batch_size=5,
            max_length=24,
            val_fraction=0.2,
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
    assert tuple(batch["input_ids"].shape) == (5, 24)
    assert tuple(batch["attention_mask"].shape) == (5, 24)
    assert tuple(batch["cuisine_labels"].shape) == (5,)
    assert tuple(batch["area_labels"].shape) == (5,)
    assert tuple(batch["party_labels"].shape) == (5,)
    assert "cuisine" in vocab.token_to_id
    assert "downtown" in vocab.token_to_id
    assert "party" in vocab.token_to_id

    model = DialogStateTracker(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_cuisine_states=4,
            num_area_states=4,
            num_party_states=4,
            embed_dim=32,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"cuisine_logits", "area_logits", "party_logits"}
    assert tuple(outputs["cuisine_logits"].shape) == (5, 4)
    assert tuple(outputs["area_logits"].shape) == (5, 4)
    assert tuple(outputs["party_logits"].shape) == (5, 4)

    loss = dialog_state_loss(
        outputs["cuisine_logits"],
        outputs["area_logits"],
        outputs["party_logits"],
        batch["cuisine_labels"],
        batch["area_labels"],
        batch["party_labels"],
    )
    assert torch.isfinite(loss)

    metrics = compute_state_metrics(
        outputs["cuisine_logits"],
        outputs["area_logits"],
        outputs["party_logits"],
        batch["cuisine_labels"],
        batch["area_labels"],
        batch["party_labels"],
    )
    assert 0.0 <= metrics["slot_acc"] <= 1.0
    assert 0.0 <= metrics["joint_goal_acc"] <= 1.0


def test_dialog_state_tracking_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_29_compact_dialog_state_tracking.data import DataConfig
    from tracks.nlp.lesson_29_compact_dialog_state_tracking.train import TrainConfig, run_training

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
                run_name="pytest_dialog_state_tracking_smoke",
                embed_dim=32,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=80,
                batch_size=8,
                max_length=24,
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
        / "lesson_29_compact_dialog_state_tracking"
        / "pytest_dialog_state_tracking_smoke"
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
