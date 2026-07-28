import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_distilled_text_classifier_batch_contract() -> None:
    from tracks.nlp.lesson_19_compact_distilled_text_classifier.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_19_compact_distilled_text_classifier.model import (
        DistilledTextClassifier,
        ModelConfig,
        classification_accuracy,
        distillation_loss,
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

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert tuple(batch["input_ids"].shape) == (6, 10)
    assert tuple(batch["attention_mask"].shape) == (6, 10)
    assert tuple(batch["labels"].shape) == (6,)

    model = DistilledTextClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            student_embed_dim=24,
            teacher_embed_dim=48,
            num_classes=4,
            dropout=0.0,
        )
    )
    outputs = model(batch)
    assert set(outputs.keys()) == {"student_logits", "teacher_logits"}
    assert tuple(outputs["student_logits"].shape) == (6, 4)
    assert tuple(outputs["teacher_logits"].shape) == (6, 4)

    loss, parts = distillation_loss(
        outputs["student_logits"], outputs["teacher_logits"], batch["labels"]
    )
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"distill_loss", "ce_loss"}
    assert float(parts["distill_loss"]) >= 0.0
    assert float(parts["ce_loss"]) >= 0.0
    acc = classification_accuracy(outputs["student_logits"], batch["labels"])
    assert 0.0 <= acc <= 1.0


def test_distilled_text_classifier_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_19_compact_distilled_text_classifier.data import DataConfig
    from tracks.nlp.lesson_19_compact_distilled_text_classifier.train import TrainConfig, run_training

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
                run_name="pytest_distilled_text_classifier_smoke",
                student_embed_dim=24,
                teacher_embed_dim=48,
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
        / "lesson_19_compact_distilled_text_classifier"
        / "pytest_distilled_text_classifier_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert metric_row["train_distill_loss"] >= 0.0
    assert metric_row["train_ce_loss"] >= 0.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_acc"] <= 1.0
