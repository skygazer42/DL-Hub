import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_dialog_escalation_risk_prediction_batch_contract() -> None:
    from tracks.nlp.lesson_39_compact_dialog_escalation_risk_prediction.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_39_compact_dialog_escalation_risk_prediction.model import (
        DialogEscalationRiskClassifier,
        ModelConfig,
        compute_accuracy,
        dialog_escalation_risk_loss,
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

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert tuple(batch["input_ids"].shape) == (6, 28)
    assert tuple(batch["attention_mask"].shape) == (6, 28)
    assert tuple(batch["labels"].shape) == (6,)
    assert batch["labels"].dtype == torch.long
    assert torch.all(batch["labels"] >= 0)
    assert torch.all(batch["labels"] < 3)
    assert "escalation" in vocab.token_to_id
    assert "risk" in vocab.token_to_id
    assert "agent" in vocab.token_to_id

    model = DialogEscalationRiskClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_risk_levels=3,
            embed_dim=32,
            dropout=0.0,
        )
    )
    logits = model(batch)
    assert tuple(logits.shape) == (6, 3)

    loss = dialog_escalation_risk_loss(logits, batch["labels"])
    assert torch.isfinite(loss)
    accuracy = compute_accuracy(logits, batch["labels"])
    assert 0.0 <= accuracy <= 1.0


def test_dialog_escalation_risk_prediction_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_39_compact_dialog_escalation_risk_prediction.data import DataConfig
    from tracks.nlp.lesson_39_compact_dialog_escalation_risk_prediction.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=19,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_dialog_escalation_risk_smoke",
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
        / "lesson_39_compact_dialog_escalation_risk_prediction"
        / "pytest_dialog_escalation_risk_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_accuracy"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_accuracy"] <= 1.0
