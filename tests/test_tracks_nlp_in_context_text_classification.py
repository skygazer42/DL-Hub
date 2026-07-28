import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_in_context_text_classification_batch_contract() -> None:
    from tracks.nlp.lesson_12_compact_in_context_text_classification.data import (
        DataConfig,
        get_dataloaders,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            num_classes=3,
            support_per_class=2,
            max_length=12,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert batch["support_input_ids"].shape == (4, 6, 12)
    assert batch["support_attention_mask"].shape == (4, 6, 12)
    assert batch["support_labels"].shape == (4, 6)
    assert batch["query_input_ids"].shape == (4, 12)
    assert batch["query_attention_mask"].shape == (4, 12)
    assert batch["query_labels"].shape == (4,)
    assert len(batch["prompt_text"]) == 4
    assert "support examples" in batch["prompt_text"][0].lower()
    assert "query:" in batch["prompt_text"][0].lower()
    assert "intent" in vocab.token_to_id


def test_in_context_text_classifier_infers_without_grad_updates() -> None:
    from tracks.nlp.lesson_12_compact_in_context_text_classification.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_12_compact_in_context_text_classification.model import (
        InContextTextClassifier,
        ModelConfig,
        classification_accuracy,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=24,
            batch_size=4,
            num_classes=3,
            support_per_class=2,
            max_length=10,
            val_fraction=0.25,
            seed=1,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    model = InContextTextClassifier(ModelConfig(vocab_size=vocab.size, pad_id=vocab.pad_id))
    outputs = model(batch)

    assert set(outputs) == {"logits", "predictions", "labels"}
    assert outputs["logits"].shape == (4, 3)
    assert outputs["predictions"].shape == (4,)
    assert outputs["labels"].shape == (4,)
    assert not any(parameter.requires_grad for parameter in model.parameters())

    acc = classification_accuracy(outputs["predictions"], outputs["labels"])
    assert 0.0 <= acc <= 1.0


def test_in_context_text_classification_eval_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_12_compact_in_context_text_classification.data import DataConfig
    from tracks.nlp.lesson_12_compact_in_context_text_classification.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                seed=7,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_in_context_text_smoke",
            ),
            DataConfig(
                num_samples=40,
                batch_size=4,
                num_classes=3,
                support_per_class=2,
                max_length=12,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_12_compact_in_context_text_classification" / "pytest_in_context_text_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert 0.0 <= metric_row["train_accuracy"] <= 1.0
    assert 0.0 <= metric_row["eval_accuracy"] <= 1.0
