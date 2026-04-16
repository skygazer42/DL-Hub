import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_dialog_sla_breach_prediction_batch_contract() -> None:
    from tracks.nlp.lesson_44_toy_dialog_sla_breach_prediction.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_44_toy_dialog_sla_breach_prediction.model import (
        DialogSlaBreachClassifier,
        ModelConfig,
        compute_accuracy,
        dialog_sla_breach_loss,
    )

    train_loader, _val_loader, vocab = get_dataloaders(
        DataConfig(
            num_samples=80,
            batch_size=8,
            max_length=28,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert tuple(batch["input_ids"].shape) == (8, 28)
    assert tuple(batch["attention_mask"].shape) == (8, 28)
    assert tuple(batch["labels"].shape) == (8,)
    assert batch["labels"].dtype == torch.long
    assert torch.all(batch["labels"] >= 0)
    assert torch.all(batch["labels"] < 2)

    for token in ("sla", "breach", "customer", "minutes"):
        assert token in vocab.token_to_id

    model = DialogSlaBreachClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_sla_levels=2,
            embed_dim=32,
            dropout=0.0,
        )
    )
    logits = model(batch)
    assert tuple(logits.shape) == (8, 2)

    loss = dialog_sla_breach_loss(logits, batch["labels"])
    assert torch.isfinite(loss)
    acc = compute_accuracy(logits, batch["labels"])
    assert 0.0 <= acc <= 1.0


def test_dialog_sla_breach_prediction_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_44_toy_dialog_sla_breach_prediction.data import DataConfig
    from tracks.nlp.lesson_44_toy_dialog_sla_breach_prediction.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=44,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_dialog_sla_breach_smoke",
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
        / "lesson_44_toy_dialog_sla_breach_prediction"
        / "pytest_dialog_sla_breach_smoke"
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

