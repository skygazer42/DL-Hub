import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_adversarial_text_classification_batch_contract() -> None:
    from tracks.nlp.lesson_20_toy_adversarial_text_classification.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_20_toy_adversarial_text_classification.model import (
        AdversarialTextClassifier,
        ModelConfig,
        classification_accuracy,
        robust_classification_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=10,
            num_classes=4,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "adversarial_input_ids",
        "adversarial_attention_mask",
        "labels",
    }
    assert tuple(batch["input_ids"].shape) == (6, 10)
    assert tuple(batch["attention_mask"].shape) == (6, 10)
    assert tuple(batch["adversarial_input_ids"].shape) == (6, 10)
    assert tuple(batch["adversarial_attention_mask"].shape) == (6, 10)
    assert tuple(batch["labels"].shape) == (6,)

    model = AdversarialTextClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=24,
            hidden_dim=32,
            num_classes=4,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"clean_logits", "adversarial_logits"}
    assert tuple(outputs["clean_logits"].shape) == (6, 4)
    assert tuple(outputs["adversarial_logits"].shape) == (6, 4)

    loss, parts = robust_classification_loss(
        outputs["clean_logits"], outputs["adversarial_logits"], batch["labels"]
    )
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"clean_ce_loss", "adv_ce_loss", "consistency_loss"}
    assert float(parts["clean_ce_loss"]) >= 0.0
    assert float(parts["adv_ce_loss"]) >= 0.0
    assert float(parts["consistency_loss"]) >= 0.0
    acc = classification_accuracy(outputs["adversarial_logits"], batch["labels"])
    assert 0.0 <= acc <= 1.0


def test_adversarial_text_classification_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_20_toy_adversarial_text_classification.data import DataConfig
    from tracks.nlp.lesson_20_toy_adversarial_text_classification.train import (
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
                run_name="pytest_adversarial_text_classification_smoke",
                embed_dim=24,
                hidden_dim=32,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=10,
                num_classes=4,
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
        / "lesson_20_toy_adversarial_text_classification"
        / "pytest_adversarial_text_classification_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_clean_acc"] <= 1.0
    assert 0.0 <= metric_row["train_adv_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_clean_acc"] <= 1.0
    assert 0.0 <= metric_row["eval_adv_acc"] <= 1.0
